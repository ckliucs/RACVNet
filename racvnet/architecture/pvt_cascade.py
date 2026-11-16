"""PVTv2-B2 encoder connected to the CASCADE decoder."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cascade_dependency import upstream_module
from .encoder_input import _prepare_encoder_input


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _default_weight_path():
    override = os.environ.get("PVT_V2_B2_WEIGHTS")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "model_data" / "pvt_v2_b2.pth"


def _unwrap_checkpoint(payload):
    if not isinstance(payload, dict):
        raise TypeError("PVTv2 checkpoint must contain a state dictionary")
    for key in ("model", "state_dict"):
        if key in payload and isinstance(payload[key], dict):
            payload = payload[key]
            break
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in payload.items()
        if torch.is_tensor(value)
    }


class PVTv2B2Encoder(nn.Module):
    """Official PVT/EMCAD PVTv2-B2 four-level feature encoder."""

    model_name = "pvt_v2_b2"
    channels = (64, 128, 320, 512)

    def __init__(self, pretrained=True, weight_path=None):
        super().__init__()
        self.backbone = upstream_module("pvtv2").pvt_v2_b2()
        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.pretrained_source = None
        self.pretrained_report = {
            "loaded_tensors": 0,
            "ignored_classification_keys": [],
            "missing_keys": [],
            "unexpected_keys": [],
        }
        if pretrained:
            self.load_pretrained(weight_path or _default_weight_path())

    def load_pretrained(self, weight_path):
        path = Path(weight_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Official PVTv2-B2 weight not found: {}. Download from "
                "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth"
                .format(path)
            )
        raw = _unwrap_checkpoint(torch.load(str(path), map_location="cpu"))
        ignored = sorted(
            name for name in raw
            if name.startswith(("head.", "classifier.", "fc."))
        )
        backbone_state = {
            name: value for name, value in raw.items() if name not in ignored
        }
        result = self.backbone.load_state_dict(backbone_state, strict=False)
        missing = sorted(result.missing_keys)
        unexpected = sorted(result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "PVTv2-B2 pretrained backbone mismatch; missing={}, unexpected={}"
                .format(missing, unexpected)
            )
        self.pretrained_source = str(path)
        self.pretrained_report = {
            "loaded_tensors": len(backbone_state),
            "ignored_classification_keys": ignored,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        return self.pretrained_report

    def forward(self, inputs):
        if inputs.dim() != 4 or inputs.size(1) != 3:
            raise ValueError("PVTv2-B2 expects Bx3xHxW")
        normalized = (inputs - self.imagenet_mean) / self.imagenet_std
        features = self.backbone(normalized)
        if len(features) != 4:
            raise RuntimeError("PVTv2-B2 must return four feature stages")
        return tuple(features)


class PVTToCASCADEAdapter(nn.Module):
    """Project the four PVTv2 feature stages to CASCADE channel widths."""

    source_channels = (64, 128, 320, 512)
    target_channels = (96, 192, 384, 768)

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(source, target, kernel_size=1, bias=True)
            for source, target in zip(self.source_channels, self.target_channels)
        ])

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("PVT-to-CASCADE adapter expects four stages")
        return tuple(layer(feature) for layer, feature in zip(self.projections, features))


class PVTv2Cascade(nn.Module):
    """PVTv2-B2 backbone, feature adapters, and CASCADE decoder."""

    pvt_channels = (64, 128, 320, 512)
    cascade_shallow_channels = (96, 192, 384, 768)
    decoder_channels = (768, 384, 192, 96)

    def __init__(self, num_classes=9, pretrained=True, deep_supervision=False,
                 weight_path=None):
        super().__init__()
        if deep_supervision:
            raise ValueError("The base segmentor does not enable auxiliary supervision")
        self.num_classes = int(num_classes)
        self.deep_supervision = False
        self.decoder_type = "cascade"
        self.pvtv2_b2 = PVTv2B2Encoder(pretrained=pretrained, weight_path=weight_path)
        self.interface_adapter = PVTToCASCADEAdapter()
        self.decoder = upstream_module("decoders").CASCADE(channels=self.decoder_channels)
        self.final = nn.Conv2d(self.decoder_channels[-1], self.num_classes, 1)
        self._last_shapes = {}

    def forward(self, inputs, return_aux=False):
        if return_aux:
            raise ValueError("The base segmentor returns the final prediction only")
        encoder_inputs = _prepare_encoder_input(inputs)
        raw = self.pvtv2_b2(encoder_inputs)
        adapted = self.interface_adapter(raw)
        stage1, stage2, stage3, stage4 = adapted
        decoder4, decoder3, decoder2, decoder1 = self.decoder(
            stage4, (stage3, stage2, stage1)
        )
        logits = self.final(decoder1)
        if logits.shape[-2:] != inputs.shape[-2:]:
            logits = F.interpolate(
                logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False
            )
        self._last_shapes = {
            "input": list(inputs.shape),
            "pvt_stage1": list(raw[0].shape),
            "pvt_stage2": list(raw[1].shape),
            "pvt_stage3": list(raw[2].shape),
            "pvt_stage4": list(raw[3].shape),
            "adapter_stage1": list(stage1.shape),
            "adapter_stage2": list(stage2.shape),
            "adapter_stage3": list(stage3.shape),
            "adapter_stage4": list(stage4.shape),
            "decoder_stage4": list(decoder4.shape),
            "decoder_stage3": list(decoder3.shape),
            "decoder_stage2": list(decoder2.shape),
            "decoder_stage1": list(decoder1.shape),
            "final_logits": list(logits.shape),
        }
        return logits

    def architecture_shapes(self):
        return dict(self._last_shapes)

    def project_module_scales_(self):
        return self

    def reset_msaff_scale_statistics(self, *args, **kwargs):
        return None

    def reset_directional_hf_statistics(self, *args, **kwargs):
        return None

    def reset_bcrd_statistics(self, *args, **kwargs):
        return None

