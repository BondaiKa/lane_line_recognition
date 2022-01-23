import numpy as np
from typing import Tuple

VIL_100_attributes = {
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    13: 10,
}

VIL_100_colour_line = {
    1: (255, 0, 0),  # single white solid
    2: (0, 255, 0),  # single white dotted
    3: (255, 125, 0),  # single yellow solid
    4: (255, 255, 0),  # single yellow dotted
    5: (255, 0, 0),  # double white solid
    6: (255, 125, 0),  # double yellow solid
    7: (255, 255, 0),  # double yellow dotted
    8: (255, 0, 0),  # double white solid dotted
    9: (255, 0, 0),  # double white dotted solid
    10: (255, 0, 0),  # double solid white and yellow
}


def get_valid_attribute(attr: int) -> int:
    return VIL_100_attributes.get(attr, attr)


def get_colour_from_one_hot_vector(vector: np.ndarray) -> Tuple[int, int, int]:
    """Get colour from one hot vector"""
    return VIL_100_colour_line.get(int(np.argmax(vector, axis=1)), None)
