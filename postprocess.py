"""Keep the largest 26-connected component for each foreground class."""

import numpy as np
from scipy import ndimage as ndi


def largest_component_26(prediction, num_classes=9):
    if prediction.ndim != 3 or not prediction.size or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("Expected a non-empty integer D,H,W label volume")
    if prediction.min() < 0 or prediction.max() >= num_classes:
        raise ValueError("Prediction contains invalid class IDs")
    result = np.zeros_like(prediction)
    structure = ndi.generate_binary_structure(3, 3)
    for class_id in range(1, num_classes):
        labels, count = ndi.label(prediction == class_id, structure=structure)
        if count:
            sizes = np.bincount(labels.ravel())
            winner = int(np.argmax(sizes[1:])) + 1
            result[labels == winner] = class_id
    return result
