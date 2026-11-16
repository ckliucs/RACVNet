"""Shared training inputs and deterministic checkpoint utilities."""

import hashlib
import random
from collections import defaultdict

import numpy as np
import torch

from .protocol import TRAIN_CASES, TRAIN_SLICES
from .synapse_slices import case_id_from_sample_name, slice_index_from_sample_name


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def train_names(data_dir, list_path):
    names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    if len(names) != TRAIN_SLICES or len(set(names)) != TRAIN_SLICES:
        raise ValueError("Expected the original ordered list of 2,211 unique training slices")
    groups = defaultdict(list)
    for name in names:
        case = case_id_from_sample_name(name)
        if not (data_dir / (name + ".npz")).is_file():
            raise FileNotFoundError(name + ".npz")
        groups[case].append(name)
    if set(groups) != set(TRAIN_CASES):
        raise ValueError("Training list does not match the fixed 18-case split")
    for names_in_case in groups.values():
        names_in_case.sort(key=slice_index_from_sample_name)
        if [slice_index_from_sample_name(name) for name in names_in_case] != list(range(len(names_in_case))):
            raise ValueError(f"Nonconsecutive slice indices for {names_in_case[0]}")
    return names, groups


def verify_sha256(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")


def save_state_dict(model, path):
    temp = path.with_suffix(".tmp")
    torch.save({name: value.cpu() for name, value in model.state_dict().items()}, temp)
    temp.replace(path)
