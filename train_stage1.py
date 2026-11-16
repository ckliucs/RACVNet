"""Train the 2.5D Synapse segmentor through the fixed 400 epochs."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import build_segmentor
from racvnet.protocol import IMAGENET_SHA256, STAGE1_SETTINGS as CONFIG
from racvnet.ram import crop_tensor
from racvnet.synapse_slices import SynapseSliceDataset
from racvnet.training import save_state_dict, seed_everything, seed_worker, train_names, verify_sha256


def segmentation_loss(logits, target):
    prediction = logits.softmax(dim=1)
    labels = F.one_hot(target, 9).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = (2 * (prediction * labels).sum(dims) + 1e-5) / (
        (prediction.square() + labels.square()).sum(dims) + 1e-5
    )
    return .5 * F.cross_entropy(logits, target) + .5 * (1 - dice.mean())


def supervised_heads(output, target):
    heads = output['deep_supervision_logits']
    losses = []
    for logits in heads:
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(segmentation_loss(logits, target))
    return (losses[0].new_tensor((1/6, 1/6, 1/6, .5)) * torch.stack(losses)).sum()


def objective(output, target):
    local_target = crop_tensor(target[:, None].float(), output["ram_boxes"], mode="nearest")[:, 0].long()
    return (.5 * supervised_heads(output, target)
            + .25 * supervised_heads(output["ram_local_output"], local_target)
            + .25 * segmentation_loss(output["seg"], target))


def update(model, optimizer, scaler, images, labels):
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    for attempt in range(12):
        if attempt:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(gpu_rng)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss = objective(model(images, return_aux=True), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite stage 1 loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(parameter.grad is None for parameter in model.parameters() if parameter.requires_grad):
            raise RuntimeError("Missing stage 1 gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if torch.isfinite(norm):
            scaler.step(optimizer)
            scaler.update()
            return float(loss.detach())
        scaler.update(new_scale=scaler.get_scale() / 2)
    raise FloatingPointError("Stage 1 AMP retry limit reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-dir", "train-list", "imagenet-weights", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("Stage 1 requires CUDA")
    names, _ = train_names(args.train_dir, args.train_list)
    verify_sha256(args.imagenet_weights, IMAGENET_SHA256)
    seed_everything(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    model = build_segmentor(args.imagenet_weights.resolve()).cuda()
    dataset = SynapseSliceDataset(args.train_dir, list_file=None, sample_names=names)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                        num_workers=CONFIG['workers'], pin_memory=True, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(CONFIG['seed']))
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = parameter.ndim == 1 or name.endswith(".bias") or "relative_position_bias_table" in name or "logit_scale" in name
        (no_decay if exempt else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": CONFIG['weight_decay']}, {"params": no_decay, "weight_decay": 0.0}],
        lr=CONFIG['lr'],
    )
    scaler = GradScaler()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    warmup = CONFIG['warmup_epochs'] * len(loader)
    steps = CONFIG['epochs'] * len(loader)
    for epoch in range(CONFIG['epochs']):
        model.train()
        total = 0.0
        for index, (images, labels, _) in enumerate(loader):
            step = epoch * len(loader) + index
            if step < warmup:
                rate = CONFIG['min_lr'] + (CONFIG['lr'] - CONFIG['min_lr']) * ((step + 1) / warmup)
            else:
                progress = (step - warmup) / (steps - warmup - 1)
                rate = CONFIG['min_lr'] + 0.5 * (CONFIG['lr'] - CONFIG['min_lr']) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = rate
            total += update(model, optimizer, scaler, images.cuda(non_blocking=True), labels.cuda(non_blocking=True))
        print(f"Stage 1 epoch {epoch+1}/{CONFIG['epochs']} loss={total / len(loader):.6f}", flush=True)
    save_state_dict(model, args.output_dir / "last_epoch_weights.pth")


if __name__ == "__main__":
    main()
