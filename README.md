# RACVNet

RACVNet is a two-stage framework for abdominal multi-organ segmentation. The first stage processes three adjacent axial CT slices with a PVTv2-B2 encoder and CASCADE decoder. The Region-Aware Module (RAM) combines global and locally refined predictions, while Context-Driven Refinement Fusion (CDRF) enhances decoder features. The second stage uses a Local Volume Refinement Module (LVRM) to refine the probability volume using CT intensities and spatial coordinates. Largest connected component (LCC) filtering is an optional post-processing step.

The repository includes the RACVNet modules, Synapse training and evaluation scripts, and checkpoint checksums.

## Installation

The original Synapse environment used Linux x86-64, Python 3.8.10, PyTorch 1.9.1+cu111, torchvision 0.10.1+cu111, CUDA 11.1, and timm 0.6.13. Install matching PyTorch wheels from the [PyTorch archive](https://pytorch.org/get-started/previous-versions/).

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt
python smoke_test.py
```

The CPU smoke test uses synthetic tensors and requires neither datasets nor checkpoints. Full training and Synapse evaluation require a CUDA GPU.

### Official CASCADE dependency

For Stage 1 training or full inference, obtain the [official CASCADE source](https://github.com/SLDGroup/CASCADE) under its [UT Austin Research License](https://github.com/SLDGroup/CASCADE/blob/d3f050ad6337ef1d8213367eefb5e141bcf2ce75/LICENSE). From the RACVNet root, set up a separate checkout:

```bash
git clone https://github.com/SLDGroup/CASCADE.git ../CASCADE
git -C ../CASCADE checkout --detach d3f050ad6337ef1d8213367eefb5e141bcf2ce75
export CASCADE_ROOT="$(cd ../CASCADE && pwd)"
python -m pip install 'opencv-python-headless==4.10.0.84' 'matplotlib==3.7.5' 'seaborn==0.13.2'
```

`CASCADE_ROOT` must contain `lib/pvtv2.py` and `lib/decoders.py`. The repository loads those files locally and attaches CDRF to the second-level attention gate (`AG2`). Standalone LVRM training and the CPU smoke test can run independently of the CASCADE checkout.

## Data preparation

Obtain the [Synapse/BTCV dataset](https://www.synapse.org/Synapse:syn3193805/wiki/217789) according to its access terms. The original evaluation uses the TransUNet-style processed arrays, an 18-case training / 12-case test split, and **2,211 ordered training slices**. The fixed case lists, class order, and training settings are defined in [`racvnet/protocol.py`](racvnet/protocol.py).

```text
Synapse/
├── train_npz/
│   └── case####_slice###.npz  # image, label: 512 x 512
└── test_vol_h5/
    └── case####.npy.h5        # image, label: D x H x W
```

Images must match the original normalized [0, 1] arrays. The 2.5D input uses `[z-1, z, z+1]` slices (edge slices are replicated), resized to 224 × 224; masks use nearest-neighbor interpolation. Evaluation restores predictions to the native voxel grid. This repository does **not** include a validated raw NIfTI-to-NPZ/H5 converter. Substituting a different CT window, orientation, or slice ordering can alter the scores.

An original ordered `train.txt` is required for training. It contains 2,211 slice stems, one per line, with SHA256 `2a3afc3b126f560d023c09e9a943e31c9ab95cea544afa69bf6b39f80ade59f5`.

```bash
export SYNAPSE=/path/to/Synapse
export SYNAPSE_TRAIN_LIST=/path/to/original/train.txt
```

## Pretrained weights

The Synapse checkpoints are available from [Google Drive](https://drive.google.com/drive/folders/1z_HkXk10TNJ7-9TnqQ6eDcef-sR8inoN?usp=drive_link).

| File | Purpose |
| --- | --- |
| `racvnet_stage1_synapse.pth` | Stage 1 segmentation network (PVTv2-B2, CASCADE, RAM, CDRF) |
| `racvnet_lvrm_synapse.pth` | Stage 2 LVRM refiner |

Place both files in `checkpoints/`. The expected SHA256 values are provided in the repository root and in `racvnet/protocol.py`.

```bash
mkdir -p checkpoints
(cd checkpoints && sha256sum -c ../SHA256SUMS.txt)
```

## Training

Download the [ImageNet-pretrained PVTv2-B2 weights](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth) before Stage 1 training.

```bash
python train_stage1.py \
  --train-dir "$SYNAPSE/train_npz" \
  --train-list "$SYNAPSE_TRAIN_LIST" \
  --imagenet-weights /path/to/pvt_v2_b2.pth \
  --output-dir runs/stage1

python train_lvrm.py \
  --train-dir "$SYNAPSE/train_npz" \
  --train-list "$SYNAPSE_TRAIN_LIST" \
  --output-dir runs/lvrm
```

Both stages use 400 epochs. LVRM is trained separately with synthetic perturbations of training annotations, CT patches, and spatial coordinates; the frozen Stage 1 model supplies probability maps at inference. The `train_lvrm.py` implementation does **not** use Stage 1 predictions during training.

## Evaluation

```bash
python test_synapse.py \
  --test-dir "$SYNAPSE/test_vol_h5" \
  --stage1-weights checkpoints/racvnet_stage1_synapse.pth \
  --lvrm-weights checkpoints/racvnet_lvrm_synapse.pth \
  --verify-paper-weights --with-lcc \
  --output-dir runs/synapse_eval
```

The default evaluation uses the fixed 12 Synapse test cases. Use `--case case0008` for a single-volume check. Without `--with-lcc`, only RACVNet is evaluated; with it, both variants are reported. The output includes prediction volumes, `per_case_organ_metrics.csv`, and `summary.csv`. The loader maps the original Stage 1 checkpoint's decoder keys to the upstream names before strict loading. It also accepts newly trained checkpoints with upstream names. Use `--verify-paper-weights` for the original paper checkpoints.

The metrics use mean Dice (%) and HD95 in **voxels**, averaged over 12 cases × 8 organs. HD95 uses the 95th percentile of pooled bidirectional surface distances on the native grid. Under the original evaluation convention, two empty masks yield Dice = 1 and HD95 = 0; if only one mask is empty, Dice = 0 and HD95 = 0. This last convention is not universal and should be considered when comparing to other implementations.

This repository provides training and evaluation scripts for Synapse. Quantitative results on Synapse, AMOS22-CT, and ACDC are reported in the manuscript. The external CASCADE integration was checked on synthetic CPU inputs for checkpoint compatibility and inference equivalence. The full Synapse evaluation with the paper checkpoints is pending for this revision.

## Acknowledgements

RACVNet builds on [PVTv2](https://github.com/whai362/PVT) and [CASCADE](https://github.com/SLDGroup/CASCADE). We also acknowledge [EMCAD](https://github.com/SLDGroup/EMCAD) and [TransUNet](https://github.com/Beckschen/TransUNet) for related implementations and data conventions.

## Citation

A paper citation will be added once the bibliographic details are available. In the meantime, reference the RACVNet repository and the commit used.

## Third-party software

Obtain the CASCADE dependency separately from the official repository and follow its upstream license. This package contains the RACVNet integration code, modules, and experiment scripts. Pretrained weights and datasets are obtained separately and remain subject to their applicable terms.
