"""Normalize and validate input tensors for the audited 2.5D encoder."""

import torch

def _prepare_encoder_input(inputs):
    """Preserve B2's strict 1/3-channel normalized input contract."""
    if inputs.dim() != 4 or inputs.size(1) not in (1, 3):
        raise ValueError(
            "Encoder expects Bx1xHxW or Bx3xHxW input, got {}".format(
                tuple(inputs.shape)
            )
        )
    if not torch.isfinite(inputs).all().item():
        raise ValueError("Encoder input contains non-finite values")
    observed_min = float(inputs.detach().min())
    observed_max = float(inputs.detach().max())
    if observed_min < -1e-4 or observed_max > 1.0001:
        raise ValueError(
            "Normalized encoder input must lie in [0,1], observed "
            "[{:.4f}, {:.4f}]".format(observed_min, observed_max)
        )
    encoder_input = inputs.clamp(0.0, 1.0)
    if encoder_input.size(1) == 1:
        encoder_input = encoder_input.repeat(1, 3, 1, 1)
    return encoder_input

