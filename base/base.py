import logging
from typing import Union
import cv2
import numpy as np
from typing import Tuple, List

log = logging.getLogger(__name__)


class MetaSingleton(type):
    """Metaclass for create singleton"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
            log.debug(f'Create new {cls}')
        return cls._instances[cls]


def calculate_perspective_transform_matrix(width: int, height: int, reverse_flag=False) -> Tuple[
    np.ndarray]:
    """
    Calculate transformation matrix for perspective transformation
    :param width: frame width
    :param height: frame height
    :param reverse_flag: create reverse matrix for reverting to initial frame
    :return: matrix for transformation the frame
    """
    # TODO @Karim: check on real Audi Q2 input frame
    # TODO: find right values for perspective transformation
    high_left_crd, high_right_crd = (550, 530), (700, 530)
    down_left_crd, down_right_crd, = (0, height - 150), (width, height - 150)

    initial_matrix = np.float32([[high_left_crd, high_right_crd,
                                  down_left_crd, down_right_crd]])
    final_matrix = np.float32([[(0, 0), (width, 0), (0, height), (width, height)]])

    return cv2.getPerspectiveTransform(initial_matrix, final_matrix) \
        if not reverse_flag else cv2.getPerspectiveTransform(final_matrix, initial_matrix)


def transform_frame(frame: np.ndarray, width: int, height: int, reverse_flag=False) -> np.ndarray:
    """
    Perform perspective transformation
    :param frame: frame
    :param width: frame width
    :param height: frame height
    :param reverse_flag: cancel perspective transformation
    :return: changed (un)transformed frame
    """
    if not reverse_flag:
        initial_matrix = calculate_perspective_transform_matrix(width, height)
        frame = cv2.warpPerspective(frame, initial_matrix, dsize=(width, height))
    else:
        final_matrix = calculate_perspective_transform_matrix(width, height, reverse_flag=True)
        frame = cv2.warpPerspective(frame, final_matrix, dsize=(width, height))
    return frame
