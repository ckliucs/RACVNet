"""Load the CASCADE components from a locally installed upstream checkout."""

import hashlib
import importlib.util
import os
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path


UPSTREAM_COMMIT = "d3f050ad6337ef1d8213367eefb5e141bcf2ce75"


def upstream_module(filename):
    root = os.environ.get("CASCADE_ROOT")
    if not root:
        raise RuntimeError(
            "Set CASCADE_ROOT to the official CASCADE checkout "
            f"at {UPSTREAM_COMMIT}. See README.md."
        )
    path = Path(root).expanduser().resolve() / "lib" / (filename + ".py")
    if not path.is_file():
        raise FileNotFoundError(f"CASCADE dependency missing: {path}")
    return _load_module(path)


@lru_cache(maxsize=None)
def _load_module(path):
    path_id = hashlib.sha256(os.fsencode(path)).hexdigest()[:16]
    name = f"_racvnet_cascade_{path.stem}_{path_id}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the CASCADE source file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        sys.modules.pop(name, None)
        raise ImportError(
            f"Could not import {path}. Install the dependencies listed in "
            f"README.md. Original error: {exc}"
        ) from exc
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_DECODER_NAMES = {
    "bottleneck_projection": "Conv_1x1",
    "spatial.projection": "SA.conv1",
}
for level in range(1, 5):
    _DECODER_NAMES.update({
        f"decoder{level}.block": f"ConvBlock{level}.conv",
        f"channel{level}.reduce": f"CA{level}.fc1",
        f"channel{level}.expand": f"CA{level}.fc2",
    })
for level in range(1, 4):
    _DECODER_NAMES.update({
        f"up{level}.block": f"Up{level}.up",
        f"attention{level}.gate_projection": f"AG{level}.W_g",
        f"attention{level}.skip_projection": f"AG{level}.W_x",
        f"attention{level}.mask": f"AG{level}.psi",
    })


def _upstream_key(key, names):
    for old, new in names.items():
        if key == old or key.startswith(old + "."):
            return new + key[len(old):]
    return key


def translate_decoder_state(state):
    """Accept original RACVNet Stage-I checkpoints alongside new ones."""
    if not isinstance(state, dict):
        raise TypeError("Expected a Stage-I state dictionary")
    names = {f"decoder.{old}": f"decoder.{new}" for old, new in _DECODER_NAMES.items()}
    translated = OrderedDict()
    for key, tensor in state.items():
        key = _upstream_key(key, names)
        if key in translated:
            raise ValueError(f"Duplicate checkpoint key after CASCADE mapping: {key}")
        translated[key] = tensor
    if hasattr(state, "_metadata"):
        translated._metadata = OrderedDict(
            (_upstream_key(key, names), version) for key, version in state._metadata.items()
        )
    return translated
