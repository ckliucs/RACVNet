import random
import re
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


SYNAPSE_SLICE_PATTERN = re.compile(r"^(case\d{4})_slice(\d+)$")

def case_id_from_sample_name(sample_name):
    """Extract a volume-level case ID from a Synapse slice name."""
    match = SYNAPSE_SLICE_PATTERN.fullmatch(str(sample_name).strip())
    if match is None:
        raise ValueError(
            "Invalid Synapse slice name {!r}; expected case####_slice###".format(
                sample_name
            )
        )
    return match.group(1)


def slice_index_from_sample_name(sample_name):
    """Extract the integer axial index from a Synapse slice name."""
    match = SYNAPSE_SLICE_PATTERN.fullmatch(str(sample_name).strip())
    if match is None:
        raise ValueError(
            "Invalid Synapse slice name {!r}; expected case####_slice###".format(
                sample_name
            )
        )
    return int(match.group(2))


def _random_rot_flip(image_stack, label):
    k = np.random.randint(0, 4)
    image_stack = np.rot90(image_stack, k, axes=(-2, -1))
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    stack_axis = axis + 1
    return (
        np.flip(image_stack, axis=stack_axis).copy(),
        np.flip(label, axis=axis).copy(),
    )


def _random_rotate(image_stack, label):
    angle = np.random.randint(-20, 20)
    image_stack = ndimage.rotate(
        image_stack,
        angle,
        axes=(-2, -1),
        order=0,
        reshape=False,
    )
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image_stack, label


class SynapseSliceDataset(Dataset):
    """Load Synapse slices and form adjacent-slice 2.5D inputs.

    Each ``.npz`` contains one normalized CT slice and its segmentation mask.
    Neighboring slices are collected in [z-1, z, z+1] order; boundary
    slices reuse the nearest available slice.
    """

    def __init__(
        self,
        data_dir="data/Synapse/train_npz",
        list_file="lists/lists_Synapse/train.txt",
        sample_names=None,
        image_size=224,
        num_classes=9,
        augment=True,
    ):
        self.data_dir = Path(data_dir)
        self.list_file = Path(list_file) if list_file is not None else None
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.augment = bool(augment)

        if not self.data_dir.is_dir():
            raise FileNotFoundError("Synapse training directory not found: {}".format(self.data_dir))
        if (self.list_file is None) == (sample_names is None):
            raise ValueError(
                "Provide exactly one of list_file or sample_names to "
                "SynapseSliceDataset"
            )

        if self.list_file is not None:
            if not self.list_file.is_file():
                raise FileNotFoundError(
                    "Synapse training list not found: {}".format(self.list_file)
                )
            self.sample_names = [
                line.strip()
                for line in self.list_file.read_text().splitlines()
                if line.strip()
            ]
            sample_source = str(self.list_file)
        else:
            if isinstance(sample_names, (str, bytes)):
                raise TypeError("sample_names must be a sequence, not a string")
            self.sample_names = [str(name).strip() for name in sample_names]
            if any(not name for name in self.sample_names):
                raise ValueError("sample_names contains an empty sample name")
            sample_source = "the provided sample_names"

        if not self.sample_names:
            raise ValueError("No samples found in {}".format(sample_source))
        if len(set(self.sample_names)) != len(self.sample_names):
            raise ValueError("Duplicate samples found in {}".format(sample_source))

        self.case_ids = tuple(
            sorted({case_id_from_sample_name(name) for name in self.sample_names})
        )

        missing = [
            name for name in self.sample_names
            if not (self.data_dir / (name + ".npz")).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "{} listed training files are missing; first missing file: {}.npz".format(
                    len(missing), missing[0]
                )
            )

    def __len__(self):
        return len(self.sample_names)

    @staticmethod
    def _validate_image(image, sample_path):
        if image.ndim != 2:
            raise ValueError(
                "Invalid image shape in {}: {}".format(sample_path, image.shape)
            )
        if not np.isfinite(image).all():
            raise ValueError("Non-finite image values in {}".format(sample_path))
        stored_min = float(image.min())
        stored_max = float(image.max())
        if stored_min < -1e-4 or stored_max > 1.0001:
            raise ValueError(
                "Normalized source {} lies outside [0,1]: [{:.4f}, {:.4f}]".format(
                    sample_path, stored_min, stored_max
                )
            )

    def __getitem__(self, index):
        sample_name = self.sample_names[index]
        sample_path = self.data_dir / (sample_name + ".npz")

        with np.load(str(sample_path)) as sample:
            center_image = np.asarray(sample["image"], dtype=np.float32)
            label = np.asarray(sample["label"])

        # 2.5D input construction:
        # use the immediately adjacent physical axial slices as channels.
        # At the first/last slice, replicate the center slice.
        case_id = case_id_from_sample_name(sample_name)
        center_index = slice_index_from_sample_name(sample_name)

        def _load_context_slice(target_index):
            candidate_name = "{}_slice{:03d}".format(case_id, target_index)
            candidate_path = self.data_dir / (candidate_name + ".npz")

            if not candidate_path.is_file():
                candidate_path = sample_path

            with np.load(str(candidate_path)) as context_sample:
                context_image = np.asarray(
                    context_sample["image"],
                    dtype=np.float32,
                )

            return context_image, candidate_path

        previous_image, previous_path = _load_context_slice(center_index - 1)
        next_image, next_path = _load_context_slice(center_index + 1)

        if (
            center_image.ndim != 2
            or label.ndim != 2
            or center_image.shape != label.shape
        ):
            raise ValueError(
                "Invalid image/label shapes in {}: {} and {}".format(
                    sample_path, center_image.shape, label.shape
                )
            )

        self._validate_image(center_image, sample_path)
        self._validate_image(previous_image, previous_path)
        self._validate_image(next_image, next_path)

        if previous_image.shape != center_image.shape:
            raise ValueError(
                "Previous/context shape mismatch for {}: {} vs {}".format(
                    sample_name,
                    previous_image.shape,
                    center_image.shape,
                )
            )

        if next_image.shape != center_image.shape:
            raise ValueError(
                "Next/context shape mismatch for {}: {} vs {}".format(
                    sample_name,
                    next_image.shape,
                    center_image.shape,
                )
            )

        if not np.isfinite(label).all():
            raise ValueError("Non-finite image or label values in {}".format(sample_path))

        # The downloaded training masks are stored as float32, but contain
        # categorical values 0..8.  Round before converting to integer labels.
        rounded_label = np.rint(label)
        if not np.allclose(label, rounded_label, rtol=0.0, atol=1e-4):
            raise ValueError(
                "Non-categorical label values found in {}".format(sample_path)
            )
        label = rounded_label.astype(np.int64)
        if label.min() < 0 or label.max() >= self.num_classes:
            raise ValueError(
                "Label values in {} fall outside [0, {}]".format(
                    sample_path, self.num_classes - 1
                )
            )

        # Channel semantics are fixed:
        # C0 = z-1, C1 = z, C2 = z+1.
        image_stack = np.stack(
            [previous_image, center_image, next_image],
            axis=0,
        )

        # Match the augmentation used by the public TransUNet Synapse loader.
        if self.augment:
            if random.random() > 0.5:
                image_stack, label = _random_rot_flip(image_stack, label)
            elif random.random() > 0.5:
                image_stack, label = _random_rotate(image_stack, label)

        height, width = image_stack.shape[-2:]
        if height != self.image_size or width != self.image_size:
            scale = (self.image_size / height, self.image_size / width)
            image_stack = np.stack(
                [
                    zoom(slice_image, scale, order=3)
                    for slice_image in image_stack
                ],
                axis=0,
            )
            label = zoom(label, scale, order=0)

        # Cubic interpolation can slightly overshoot a bounded source.
        image_stack = np.clip(image_stack, 0.0, 1.0)
        image_stack = image_stack.astype(np.float32, copy=False)
        label = label.astype(np.int64, copy=False)

        # Already three physical axial slices; do not repeat center slice.

        return (
            torch.from_numpy(np.ascontiguousarray(image_stack)),
            torch.from_numpy(np.ascontiguousarray(label)),
            sample_name,
        )

