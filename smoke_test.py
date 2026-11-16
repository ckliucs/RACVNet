"""Run small, synthetic CPU checks for the three standalone components."""

import argparse

import numpy as np
import torch

from postprocess import largest_component_26
from racvnet.metrics import metrics
from racvnet.cdrf import ContextDrivenRefinementFusion
from racvnet.lvrm import Refiner
from racvnet.ram import crop_tensor, fuse_logits, propose_boxes


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    with torch.no_grad():
        logits = torch.zeros(1, 9, 224, 224)
        logits[:, 0] = 12
        image = torch.rand(1, 3, 224, 224)
        boxes = propose_boxes(logits)
        torch.testing.assert_close(boxes, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
        torch.testing.assert_close(crop_tensor(image, boxes), image, atol=1e-4, rtol=1e-4)
        fused, alpha = fuse_logits(logits, logits, boxes)
        torch.testing.assert_close(fused, logits, atol=1e-5, rtol=1e-5)
        assert alpha.shape == (1, 1, 224, 224) and alpha.max() == 0.5

        semantic, detail = (torch.rand(1, 192, 28, 28) for _ in range(2))
        result = ContextDrivenRefinementFusion().eval()(semantic, detail, detail)
        assert result.shape == detail.shape and torch.isfinite(result).all()

        probability = torch.rand(1, 9, 3, 16, 16).softmax(1)
        volume = torch.rand(1, 1, 3, 16, 16)
        coordinates = torch.zeros(1, 3, 3, 16, 16)
        refined = Refiner().eval()(probability, volume, coordinates)
        assert torch.equal(refined, probability.clamp_min(1e-6).log())

    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0, 0, 0] = mask[1, 1, 1] = mask[4, 4, 4] = 1
    kept = largest_component_26(mask)
    assert kept[1, 1, 1] == 1 and kept[4, 4, 4] == 0
    empty = np.zeros_like(mask, dtype=bool)
    assert metrics(empty, empty)[0]["dice"] == 1
    assert metrics(kept == 1, empty)[0]["dice"] == 0
    assert metrics(kept == 1, empty)[0]["hd95_voxel"] == 0
    print("PASS: RAM, CDRF, LVRM, 26-neighbor LCC and empty-mask metrics (CPU)")


if __name__ == "__main__":
    main()
