"""Evaluate the fixed 12 Synapse cases in the original voxel grid."""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch

from model import RACVNet, load_weights
from postprocess import largest_component_26
from racvnet.metrics import metrics
from racvnet.protocol import LVRM_SHA256, ORGANS, STAGE1_SHA256, TEST_CASES
from racvnet.training import verify_sha256


def evaluate(model, test_dir, output_dir, with_lcc, cases=TEST_CASES):
    paths = {case: test_dir / (case + ".npy.h5") for case in cases}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fixed test volumes: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, path in paths.items():
        with h5py.File(path, "r") as sample:
            image = sample["image"][:]
        raw = model.predict_volume(image)
        predictions = {"RACVNet": raw}
        if with_lcc:
            predictions["RACVNet+LCC"] = largest_component_26(raw)
        for name, prediction in predictions.items():
            folder = output_dir / name
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / (case + "_prediction.npz"), prediction=prediction)
        with h5py.File(path, "r") as sample:
            target = sample["label"][:]
        if target.shape != raw.shape or not np.isin(target, np.arange(9)).all():
            raise ValueError(f"Invalid ground-truth grid or labels in {path}")
        for name, prediction in predictions.items():
            for index, organ in enumerate(ORGANS, 1):
                values, _ = metrics(prediction == index, target == index)
                rows.append(dict(case=case, method=name, class_id=index, organ=organ, **values))
        print(f"Evaluated {case}", flush=True)
    with (output_dir / "per_case_organ_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "organ", "dice_pct", "hd95_voxel", "records", "empty_hd95_records"))
        for name in sorted({row['method'] for row in rows}):
            for organ in ("mean", *ORGANS):
                group = [row for row in rows if row['method'] == name and (organ == "mean" or row['organ'] == organ)]
                dice = 100 * np.mean([row['dice'] for row in group])
                hd95 = np.mean([row['hd95_voxel'] for row in group])
                empty = sum(row['empty_hd95_bookkeeping'] for row in group)
                writer.writerow((name, organ, dice, hd95, len(group), empty))
                if organ == "mean":
                    print(f"{name}: DSC={dice:.4f}% HD95={hd95:.4f} voxel, n={len(group)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("test-dir", "stage1-weights", "lvrm-weights", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--with-lcc", action="store_true", help="Report both RACVNet and RACVNet + LCC")
    parser.add_argument("--case", choices=TEST_CASES, help="Run a single fixed test case for a smoke check")
    parser.add_argument("--verify-paper-weights", action="store_true",
                        help="Require both exact paper checkpoint hashes")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Evaluation of the paper configuration requires CUDA")
    if args.verify_paper_weights:
        verify_sha256(args.stage1_weights, STAGE1_SHA256)
        verify_sha256(args.lvrm_weights, LVRM_SHA256)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = RACVNet()
    load_weights(model.segmentor, args.stage1_weights)
    load_weights(model.refiner, args.lvrm_weights)
    print("Both checkpoints loaded with strict=True. No missing or unexpected keys.", flush=True)
    evaluate(model.cuda().eval(), args.test_dir, args.output_dir, args.with_lcc,
             (args.case,) if args.case else TEST_CASES)


if __name__ == "__main__":
    main()
