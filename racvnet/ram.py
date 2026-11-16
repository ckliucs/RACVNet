"""Image-predicted boxes,224->ROI224 resampling,and inverse-grid fusion.

Coordinates are normalized pixel EDGES [x0,y0,x1,y1], align_corners=False.
This reallocates tokens within existing224 inputs; it does not recover native
CT detail discarded by the original resize. No GT/patient/phase IDs used here.
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

IMAGE_SIZE = 224
MIN_SIDE = 80.
MASS_THRESHOLD = 16.
QUANTILE = .02
FEATHER_PIXELS = 8.
MAX_LOCAL_WEIGHT = .5


@torch.no_grad()
def propose_boxes(logits):
    if logits.ndim != 4 or logits.shape[1:] != (9, 224, 224):
        raise ValueError("ROI proposals require nine-class224 image predictions")
    with autocast(enabled=False):
        foreground = 1. - logits.detach().float().softmax(1)[:, 0]
        mass = (foreground - .5).clamp_min(0.)
        total = mass.sum((1, 2))
        x, y = mass.sum(1), mass.sum(2)
        cdfs = [v.cumsum(1) / total[:, None].clamp_min(1e-12) for v in (x, y)]
        low, high = [], []
        for cdf in cdfs:
            qlow = torch.full((len(cdf), 1), QUANTILE, device=logits.device)
            qhigh = torch.full_like(qlow, 1. - QUANTILE)
            low.append(torch.searchsorted(cdf.contiguous(), qlow).squeeze(1).clamp(0, 223).float())
            high.append(torch.searchsorted(cdf.contiguous(), qhigh).squeeze(1).clamp(0, 223).float())
        dimension = torch.maximum(high[0] - low[0] + 1., high[1] - low[1] + 1.)
        # SOPSeg Eq1 adapted to224; clamp min80 to avoid extreme zoom/noise.
        k = (224. - 2. * 32.) / (224. - 32.)
        side = torch.where(dimension < 32., 2. * dimension,
                           k * dimension + (2. - k) * 32.).clamp(MIN_SIDE, 224.)
        centers = [(a + b + 1.) / 2. for a, b in zip(low, high)]
        centers = [torch.minimum(torch.maximum(c, side / 2.), 224. - side / 2.) for c in centers]
        boxes = torch.stack((centers[0] - side / 2., centers[1] - side / 2.,
                             centers[0] + side / 2., centers[1] + side / 2.), 1) / 224.
        full = torch.tensor([0., 0., 1., 1.], device=logits.device).expand_as(boxes)
        return torch.where((total >= MASS_THRESHOLD)[:, None], boxes, full)


def validate_boxes(boxes, batch):
    if boxes.shape != (batch, 4) or not torch.isfinite(boxes).all():
        raise ValueError("Require finite Bx4 ROI boxes")
    if not ((boxes >= 0).all() and (boxes <= 1).all() and
            (boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError("ROI boxes must lie inside the image with positive area")


def crop_tensor(value, boxes, mode="bilinear", output_size=224):
    validate_boxes(boxes, len(value))
    with autocast(enabled=False):
        b = boxes.detach().float()
        theta = torch.zeros((len(value), 2, 3), dtype=torch.float32, device=value.device)
        theta[:, 0, 0], theta[:, 1, 1] = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        theta[:, 0, 2], theta[:, 1, 2] = b[:, 2] + b[:, 0] - 1., b[:, 3] + b[:, 1] - 1.
        grid = F.affine_grid(theta, (len(value), value.shape[1], output_size, output_size), align_corners=False)
        return F.grid_sample(value.float(), grid, mode=mode, padding_mode="border", align_corners=False)


def inverse_roi(value, boxes, canvas_size=224):
    validate_boxes(boxes, len(value))
    with autocast(enabled=False):
        b = boxes.detach().float()
        axis = (torch.arange(canvas_size, device=value.device, dtype=torch.float32) + .5) / canvas_size
        # Explicit indexing is compatible with Torch1.9 (no indexing= argument).
        yy, xx = torch.meshgrid(axis, axis)
        xx, yy = xx[None], yy[None]
        gx = 2. * (xx - b[:, 0, None, None]) / (b[:, 2] - b[:, 0])[:, None, None] - 1.
        gy = 2. * (yy - b[:, 1, None, None]) / (b[:, 3] - b[:, 1])[:, None, None] - 1.
        grid = torch.stack((gx.expand(-1, canvas_size, -1), gy.expand(-1, -1, canvas_size)), -1)
        restored = F.grid_sample(value.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
        distances = torch.stack(((xx - b[:, 0, None, None]).expand(-1, canvas_size, -1),
                                 (b[:, 2, None, None] - xx).expand(-1, canvas_size, -1),
                                 (yy - b[:, 1, None, None]).expand(-1, -1, canvas_size),
                                 (b[:, 3, None, None] - yy).expand(-1, -1, canvas_size)))
        feather = (distances.min(0).values * canvas_size / FEATHER_PIXELS).clamp(0., 1.)[:, None]
        return restored, MAX_LOCAL_WEIGHT * feather


def fuse_logits(global_logits, local_logits, boxes):
    with autocast(enabled=False):
        warped, alpha = inverse_roi(local_logits, boxes, global_logits.shape[-1])
        gp, lp = global_logits.float().log_softmax(1), warped.log_softmax(1)
        log_alpha = torch.where(alpha > 0., alpha.clamp_min(1e-30).log(), torch.full_like(alpha, float("-inf")))
        mixture = torch.logaddexp(gp + torch.log1p(-alpha), lp + log_alpha)
        fused = global_logits.float() + (mixture - gp)
        return fused, alpha
