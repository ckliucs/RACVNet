"""Train the Synapse LVRM for 400 epochs on synthetic label perturbations."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from racvnet.lvrm import Refiner, objective
from racvnet.protocol import IMAGE_SIZE, LVRM_SETTINGS as CONFIG
from racvnet.synthetic_volumes import SyntheticVolumes
from racvnet.training import save_state_dict, seed_everything, train_names


def prepare_cache(data_dir, groups, cache_dir):
    cache_dir.mkdir()
    for case, names in groups.items():
        images, masks = [], []
        for name in names:
            with np.load(data_dir / (name + ".npz"), allow_pickle=False) as sample:
                image, mask = sample['image'], sample['label']
            if image.shape != (512, 512) or image.shape != mask.shape or not np.isfinite(image).all():
                raise ValueError(f"Invalid training slice {name}")
            if image.min() < -1e-4 or image.max() > 1.0001 or not np.isin(mask, np.arange(9)).all():
                raise ValueError(f"Training values out of range in {name}")
            images.append(np.clip(zoom(image, (IMAGE_SIZE/512, IMAGE_SIZE/512), order=3), 0, 1).astype(np.float32))
            masks.append(zoom(mask, (IMAGE_SIZE/512, IMAGE_SIZE/512), order=0).astype(np.uint8))
        np.savez_compressed(cache_dir / (case + ".npz"), image=np.stack(images), label=np.stack(masks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-dir", "train-list", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("Synapse LVRM training requires CUDA")
    _, groups = train_names(args.train_dir, args.train_list)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = args.output_dir / "train_cache"
    prepare_cache(args.train_dir, groups, cache_dir)
    seed_everything(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(4)
    model = Refiner().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    for epoch in range(1, CONFIG['epochs'] + 1):
        warmup = CONFIG['warmup']
        fraction = epoch / warmup if epoch <= warmup else .5 * (1 + math.cos(math.pi * (epoch - warmup) / (CONFIG['epochs'] - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = CONFIG['min_lr'] + (CONFIG['lr'] - CONFIG['min_lr']) * fraction
        loader = DataLoader(SyntheticVolumes(epoch, cache_dir), batch_size=CONFIG['batch_size'], shuffle=True,
                            num_workers=CONFIG['workers'], pin_memory=True,
                            generator=torch.Generator().manual_seed(CONFIG['seed'] + epoch))
        model.train()
        total = 0.0
        for sample in loader:
            probability, image, coordinates, target = (sample[key].cuda() for key in ('prob', 'image', 'coord', 'target'))
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective(model(probability, image, coordinates), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite LVRM loss")
            loss.backward()
            if any(parameter.grad is None for parameter in model.parameters()):
                raise RuntimeError("Missing LVRM gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip'])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite LVRM gradient")
            optimizer.step()
            total += float(loss.detach())
        print(f"LVRM epoch {epoch}/{CONFIG['epochs']} loss={total / len(loader):.6f}", flush=True)
    save_state_dict(model, args.output_dir / "last_epoch_weights.pth")


if __name__ == "__main__":
    main()
