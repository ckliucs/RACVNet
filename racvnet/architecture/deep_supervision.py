"""PVTv2-B2/CASCADE segmentation with auxiliary training heads.

Three 1x1 prediction heads provide deep supervision at decoder stages 4, 3,
and 2. Inference uses the final main prediction head only.
"""

import torch.nn as nn
import torch.nn.functional as F

from .pvt_cascade import PVTv2Cascade
from .encoder_input import _prepare_encoder_input


HEAD_NAMES = ("Head1", "Head2", "Head3", "Main")


class PVTv2CascadeDeepSupervision(PVTv2Cascade):
    """Add three auxiliary 1x1 prediction projections."""

    def __init__(self, num_classes=9, pretrained=True, weight_path=None):
        # Preserve the original initialization order; auxiliary heads are added last.
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            deep_supervision=False,
            weight_path=weight_path,
        )
        self.deep_supervision = True
        self.auxiliary_heads = nn.ModuleList(
            [
                nn.Conv2d(self.decoder_channels[0], self.num_classes, 1),
                nn.Conv2d(self.decoder_channels[1], self.num_classes, 1),
                nn.Conv2d(self.decoder_channels[2], self.num_classes, 1),
            ]
        )
        self._last_supervision_shapes = {}

    def forward(self, inputs, return_aux=False):
        encoder_inputs = _prepare_encoder_input(inputs)
        raw = self.pvtv2_b2(encoder_inputs)
        adapted = self.interface_adapter(raw)
        stage1, stage2, stage3, stage4 = adapted
        decoder4, decoder3, decoder2, decoder1 = self.decoder(
            stage4, (stage3, stage2, stage1)
        )

        # The inherited final head remains the main prediction head.
        main_raw = self.final(decoder1)
        main_logits = main_raw
        if main_logits.shape[-2:] != inputs.shape[-2:]:
            main_logits = F.interpolate(
                main_logits,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
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
            "final_logits": list(main_logits.shape),
        }

        if not return_aux:
            # Inference uses only the main head and the same bilinear resizing.
            return main_logits

        auxiliary_raw = tuple(
            head(feature)
            for head, feature in zip(
                self.auxiliary_heads, (decoder4, decoder3, decoder2)
            )
        )
        supervision_raw = auxiliary_raw + (main_raw,)
        self._last_supervision_shapes = {
            name: list(logits.shape)
            for name, logits in zip(HEAD_NAMES, supervision_raw)
        }
        return {
            "seg": main_logits,
            "deep_supervision_logits": supervision_raw,
            "deep_supervision_names": HEAD_NAMES,
        }

    def supervision_shapes(self):
        return dict(self._last_supervision_shapes)


