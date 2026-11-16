"""Assemble RACVNet from the local Stage-I architecture and LVRM."""


import numpy as np
import torch
from scipy.ndimage import zoom
from torch import nn

from postprocess import largest_component_26
from racvnet.lvrm import Refiner
from racvnet.protocol import IMAGE_SIZE, STAGE1_SETTINGS
from racvnet.synthetic_volumes import coordinates


def build_segmentor(imagenet_weights=None):
    """Construct the audited Stage-I network using local model definitions."""
    from racvnet.architecture.segmentor import build as build_stage1
    return build_stage1(
        str(imagenet_weights) if imagenet_weights is not None else None,
        pretrained=imagenet_weights is not None,
    )


def load_weights(module, path):
    from racvnet.architecture.segmentor import RACVNetSegmentor

    state = torch.load(path, map_location="cpu")
    if isinstance(module, RACVNetSegmentor):
        module.load_state_dict(state, strict=True)
        return
    expected = module.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError(f"Checkpoint keys do not match the model: {path}")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or (tensor.shape, tensor.dtype) != (expected[name].shape, expected[name].dtype):
            raise ValueError(f"Invalid tensor {name} in {path}")
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Strict load returned incompatible keys for {path}")


def resize_image(volume):
    scale = tuple(IMAGE_SIZE / length for length in volume.shape[1:])
    return np.stack([np.clip(zoom(image, scale, order=3), 0, 1) for image in volume]).astype(np.float32)


def restore_mask(mask, native_hw):
    scale = tuple(length / IMAGE_SIZE for length in native_hw)
    return np.stack([zoom(plane, scale, order=0) for plane in mask]).astype(np.uint8)


class RACVNet(nn.Module):
    """Synapse segmentor, trained volumetric refiner and optional LCC inference."""

    def __init__(self):
        super().__init__()
        self.segmentor = build_segmentor()
        self.refiner = Refiner()

    def forward(self, triplets):
        return self.segmentor(triplets)

    @torch.no_grad()
    def predict_volume(self, native_image, use_lcc=False):
        if self.training:
            raise RuntimeError("Set RACVNet to evaluation mode first")
        if native_image.ndim != 3 or not native_image.size or not np.isfinite(native_image).all():
            raise ValueError("Expected a finite D,H,W volume")
        if native_image.min() < -1e-4 or native_image.max() > 1.0001:
            raise ValueError("Expected the normalized Synapse H5 image")
        device = next(self.parameters()).device
        image = resize_image(np.asarray(native_image, dtype=np.float32))
        step = STAGE1_SETTINGS['batch_size']
        probabilities = []
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=True):
            for start in range(0, len(image), step):
                z = np.arange(start, min(start + step, len(image)))
                indices = np.clip(z[:, None] + np.array([-1, 0, 1]), 0, len(image) - 1)
                logits = self(torch.from_numpy(image[indices]).to(device)).float()
                if logits.shape != (len(z), 9, IMAGE_SIZE, IMAGE_SIZE) or not torch.isfinite(logits).all():
                    raise FloatingPointError("Invalid segmentation logits")
                probabilities.append(logits.softmax(1).cpu().numpy())
        probability = np.concatenate(probabilities).transpose(1, 0, 2, 3)
        position = coordinates(image.shape)
        refined = np.empty(image.shape, dtype=np.uint8)
        for start in range(0, len(image), 16):
            stop = min(start + 16, len(image))
            left, right = max(0, start - 8), min(len(image), stop + 8)
            p = torch.from_numpy(probability[:, left:right][None]).to(device)
            x = torch.from_numpy(image[left:right][None, None]).to(device)
            c = torch.from_numpy(position[:, left:right][None]).to(device)
            logits = self.refiner(p, x, c)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite volumetric logits")
            refined[start:stop] = logits.argmax(1)[0, start-left:stop-left].cpu().numpy()
        native = restore_mask(refined, native_image.shape[1:])
        return largest_component_26(native) if use_lcc else native
