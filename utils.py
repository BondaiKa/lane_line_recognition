from typing import List
import numpy as np


def one_hot_list_encoder(target_class_idx: int, num_classes: int) -> np.ndarray:
    """One-hot list encoder"""
    target_vector = np.zeros(num_classes)
    target_vector[target_class_idx] = 1
    return target_vector
