import logging
from abc import ABC
from typing import Union
import cv2
import numpy as np
from typing import NamedTuple, Tuple, List

log = logging.getLogger(__name__)

LABELS = {
    0: (0, 255, 0),  # green
    1: (255, 0, 0),  # red
}


class MetaSingleton(type):
    """Metaclass for create singleton"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
            log.debug(f'Create new {cls}')
        return cls._instances[cls]


class AbstractVideoHandler(ABC):
    """Base Video handler"""

    def __init__(self, camera_path: Union[str, int]):
        self.video = cv2.VideoCapture(camera_path)

    def process(self):
        """start to capture video"""
        raise NotImplemented

    def release(self):
        """release camera"""
        self.video.release()

    def exec(self):
        log.info("Start video processing...")
        self.process()
        log.info("End video processing...")


# TODO @Karim: remove after debugging perspective_transformation
def draw_sequence_in_img(frame: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = thickness = 3
    lineType = 2
    frame_points = frame.copy()
    for num, point in enumerate(points):
        frame_points = cv2.putText(frame_points, str(num + 1), point, font, fontScale, color, thickness,
                                   lineType)
    cv2.imshow(f"Frame_with_points_{color[0]}_{color[1]}_{color[2]}", frame_points)


def calculate_perspective_transform_matrix(frame: np.ndarray, width: int, height: int, reverse_flag=False) -> Tuple[
    np.ndarray]:
    """
    Calculate transofmration matrix for perspective transformation
    :param frame: one readed frame in video flow
    :param width: frame width
    :param height: frame height
    :param reverse_flag: create reverse matrix for reverting to initial frame
    :return: matrix for transformation the frame
    """
    high_left_crd, high_right_crd = (550, 530), (700, 530)
    down_left_crd, down_right_crd, = (0, height - 150), (width, height - 150)

    initial_matrix = np.float32([[high_left_crd, high_right_crd,
                                  down_left_crd, down_right_crd]])
    final_matrix = np.float32([[(0, 0), (width, 0), (0, height), (width, height)]])

    return cv2.getPerspectiveTransform(initial_matrix, final_matrix) \
        if not reverse_flag else cv2.getPerspectiveTransform(final_matrix, initial_matrix)


class FrameHandler(metaclass=MetaSingleton):
    labels = LABELS

    def __init__(self):
        ###
        # TODO @Karim: load nn weights etc
        ###
        ...

    @staticmethod
    def preprocess_frame(frame: np.ndarray, width, height):
        ###
        # TODO @Karim: add perspective transformation
        ###

        ###
        # TODO @Karim: apply filter
        ###
        log.debug(f"Before resizing: width {width}, Height {height}, Frame shape:{frame.shape}")
        frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
        # log.debug(f"After resizing {frame.shape}")
        # cv2.imshow('Resized frame', frame)
        initial_matrix = calculate_perspective_transform_matrix(frame, width, height)
        presp_frame = cv2.warpPerspective(frame, initial_matrix, dsize=(width, height))
        cv2.imshow('Perspective_tranform_frame', presp_frame)
        final_matrix = calculate_perspective_transform_matrix(frame, width, height, reverse_flag=True)
        reversed_frame = cv2.warpPerspective(frame, final_matrix, dsize=(width, height))
        return reversed_frame

    @staticmethod
    def recognize(frame: np.ndarray) -> np.ndarray:
        ###
        # TODO @Karim: add process image by NN
        ###
        ...

    @classmethod
    def get_colour(cls, labels):
        # TODO @Karim: transform number to colour
        ...

    @classmethod
    def draw_popylines(cls, frame: np.ndarray, points, labels: np.ndarray):
        ...
        ###
        # TODO @Karim: try to understand `PoseArray` and further logic
        ###
        colour = cls.get_colour()
        # TODO @Karim: revert perspective points to normal
        return np.apply_along_axis(cv2.polylines, axis=1, arr=points,
                                   img=frame, isClosed=True, color=colour, thickness=2)
