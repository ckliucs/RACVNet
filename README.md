# RACVNet

**Volumetric Medical Image Segmentation with Region Awareness and Contextual Feature Refinement**

RACVNet is a two-stage framework for medical image segmentation. **Stage I** performs 2.5D segmentation using three adjacent slices, combining full-slice and region-of-interest (ROI) predictions through the Region Aware Module (RAM) and Context Driven Refinement Fusion (CDRF). **Stage II** uses the Local Volume Refinement Module (LVRM) to refine the assembled probability volume.

This repository provides model components and training and evaluation scripts for **Synapse**.

## Installation

The original Synapse setup uses **Linux x86-64, Python 3.8.10, PyTorch 1.9.1+cu111, torchvision 0.10.1+cu111, CUDA 11.1, and timm 0.6.13**. Install a matching PyTorch build from the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/).

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt
python smoke_test.py
```

### CASCADE dependency

Stage I training and full inference use the original [CASCADE](https://github.com/SLDGroup/CASCADE) implementation. From the RACVNet repository root:

```bash
git clone https://github.com/SLDGroup/CASCADE.git ../CASCADE
git -C ../CASCADE checkout --detach d3f050ad6337ef1d8213367eefb5e141bcf2ce75
export CASCADE_ROOT="$(cd ../CASCADE && pwd)"
python -m pip install 'opencv-python-headless==4.10.0.84' \
  'matplotlib==3.7.5' 'seaborn==0.13.2'
```

`CASCADE_ROOT` should point to a directory containing `lib/pvtv2.py` and `lib/decoders.py`.

## Dataset Preparation

Download [Synapse/BTCV](https://www.synapse.org/Synapse:syn3193805/wiki/217789). The scripts expect **TransUNet-style processed arrays**, rather than raw NIfTI volumes:

```text
Synapse/
├── train_npz/
│   └── case####_slice###.npz   # keys: image, label; 512 x 512
└── test_vol_h5/
    └── case####.npy.h5        # datasets: image, label; D x H x W
```

- **Data split:** 18 training scans and 12 test scans; case IDs and the eight foreground classes are defined in [`racvnet/protocol.py`](racvnet/protocol.py).
- **Training list:** the ordered `train.txt` contains 2,211 slice stems, one per line. The expected SHA256 is `2a3afc3b126f560d023c09e9a943e31c9ab95cea544afa69bf6b39f80ade59f5`.
- **Input:** three neighboring slices `[z-1, z, z+1]` with edge replication, resized to 224 × 224. Labels use nearest-neighbor interpolation.
- **Preprocessing:** processed images should follow the original normalized `[0, 1]` convention. Obtain or prepare the processed arrays and the ordered training list before running the scripts.

Set the local paths:

```bash
export SYNAPSE=/path/to/Synapse
export SYNAPSE_TRAIN_LIST=/path/to/original/train.txt
```

## Pretrained Weights

Download the Synapse checkpoints from [Google Drive](https://drive.google.com/drive/folders/1z_HkXk10TNJ7-9TnqQ6eDcef-sR8inoN?usp=drive_link):

| Checkpoint | Description |
| --- | --- |
| `racvnet_stage1_synapse.pth` | Stage I: PVTv2-B2, CASCADE, RAM, and CDRF |
| `racvnet_lvrm_synapse.pth` | Stage II: LVRM |

Place both files in `checkpoints/` and verify their SHA256 checksums:

```bash
mkdir -p checkpoints
# Copy the downloaded .pth files into checkpoints/
(cd checkpoints && sha256sum -c ../SHA256SUMS.txt)
```

## Inference and Evaluation

Evaluate the fixed 12-case Synapse test split without largest connected component (LCC) filtering:

```bash
python test_synapse.py \
  --test-dir "$SYNAPSE/test_vol_h5" \
  --stage1-weights checkpoints/racvnet_stage1_synapse.pth \
  --lvrm-weights checkpoints/racvnet_lvrm_synapse.pth \
  --verify-paper-weights \
  --output-dir runs/synapse_eval
```

To evaluate the refined prediction with and without LCC filtering in the same run:

```bash
python test_synapse.py \
  --test-dir "$SYNAPSE/test_vol_h5" \
  --stage1-weights checkpoints/racvnet_stage1_synapse.pth \
  --lvrm-weights checkpoints/racvnet_lvrm_synapse.pth \
  --verify-paper-weights --with-lcc \
  --output-dir runs/synapse_eval_lcc
```

To run a single case, add `--case case0008` and choose a new output directory. The script saves prediction volumes, `per_case_organ_metrics.csv`, and `summary.csv`.

**Metric conventions:** DSC is reported as a percentage; Synapse HD95 uses voxel coordinates. Optional LCC retains the largest 26-connected component for each foreground class. See [`racvnet/metrics.py`](racvnet/metrics.py) and [`postprocess.py`](postprocess.py) for the implementations.

## Training

Download the [ImageNet-pretrained PVTv2-B2 weights](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth) for Stage I training.

**Stage I — 2.5D segmentation**

```bash
python train_stage1.py \
  --train-dir "$SYNAPSE/train_npz" \
  --train-list "$SYNAPSE_TRAIN_LIST" \
  --imagenet-weights /path/to/pvt_v2_b2.pth \
  --output-dir runs/stage1
```

**Stage II — LVRM**

```bash
python train_lvrm.py \
  --train-dir "$SYNAPSE/train_npz" \
  --train-list "$SYNAPSE_TRAIN_LIST" \
  --output-dir runs/lvrm
```

Both training scripts use settings in [`racvnet/protocol.py`](racvnet/protocol.py). Stage II is trained separately on synthetic perturbations of the training annotations.

## Project Structure

```text
RACVNet-main/
├── README.md
├── requirements.txt
├── SHA256SUMS.txt
├── model.py
├── train_stage1.py
├── train_lvrm.py
├── test_synapse.py
├── smoke_test.py
├── postprocess.py
└── racvnet/
    ├── architecture/
    ├── protocol.py
    ├── synapse_slices.py
    ├── synthetic_volumes.py
    ├── ram.py
    ├── cdrf.py
    ├── lvrm.py
    ├── metrics.py
    └── training.py
```

## Acknowledgements

The implementation builds on [PVTv2](https://github.com/whai362/PVT) and [CASCADE](https://github.com/SLDGroup/CASCADE). We also thank the authors of [EMCAD](https://github.com/SLDGroup/EMCAD) and [TransUNet](https://github.com/Beckschen/TransUNet) for their publicly available implementations.
