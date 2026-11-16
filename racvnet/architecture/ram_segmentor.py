"""Region-aware segmentation with shared weights and separate BatchNorm statistics.

The global and local branches share trainable parameters. Running statistics
are maintained separately for the local pass.
"""
from contextlib import contextmanager
import torch
from torch import nn
from .deep_supervision import PVTv2CascadeDeepSupervision
from racvnet.ram import propose_boxes, crop_tensor, fuse_logits


class RunningStats(nn.Module):
    def __init__(self, batchnorm):
        super().__init__()
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            self.register_buffer(name, getattr(batchnorm, name).detach().clone())


class RegionAwareSegmentor(PVTv2CascadeDeepSupervision):
    def __init__(self, weights, pretrained=False):
        super().__init__(num_classes=9, pretrained=pretrained, weight_path=weights)
        if self.decoder_type != "cascade" or tuple(self.decoder_channels) != (768, 384, 192, 96):
            raise ValueError("RAM requires the configured CASCADE decoder")
        layers = [(n, m) for n, m in self.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
        self._ram_bn_names = tuple(n for n, _ in layers)
        self.ram_local_stats = nn.ModuleDict({"bn%03d" % i: RunningStats(m) for i, (_, m) in enumerate(layers)})
        if not layers:
            raise RuntimeError("RAM requires BatchNorm running statistics")

    @contextmanager
    def local_statistics(self):
        modules, old = dict(self.named_modules()), []
        names = ("running_mean", "running_var", "num_batches_tracked")
        try:
            for i, name in enumerate(self._ram_bn_names):
                module, holder = modules[name], self.ram_local_stats["bn%03d" % i]
                saved = tuple(getattr(module, n) for n in names)
                old.append((module, saved, holder))
                for n in names:
                    setattr(module, n, getattr(holder, n))
            yield
        finally:
            for module, saved, holder in old:
                for n, value in zip(names, saved):
                    # Torch1.9 rebinds num_batches_tracked instead of mutating
                    # it in-place. Commit the current local tensor references
                    # before restoring global ones (also valid on newer Torch).
                    setattr(holder, n, getattr(module, n))
                    setattr(module, n, value)

    def global_only(self, images, return_aux=False):
        """Run the full-image branch without local refinement."""
        return super().forward(images, return_aux=return_aux)

    def forward(self, images, return_aux=False):
        if images.shape[1:] != (3, 224, 224):
            raise ValueError("RAM requires three-channel 224x224 inputs")
        global_output = super().forward(images, return_aux=True)
        boxes = propose_boxes(global_output["seg"])
        local_images = crop_tensor(images, boxes)
        with self.local_statistics():
            local_output = super().forward(local_images, return_aux=True)
        final, alpha = fuse_logits(global_output["seg"], local_output["seg"], boxes)
        if not return_aux:
            return final
        return dict(global_output, seg=final, ram_global_seg=global_output["seg"],
                    ram_boxes=boxes, ram_local_output=local_output, ram_alpha=alpha)


def build(weights, pretrained=False):
    return RegionAwareSegmentor(weights, pretrained=pretrained)
