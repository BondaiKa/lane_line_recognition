import logging
from abc import ABC
from typing import Union
import cv2
import numpy as np
from typing import NamedTuple, Tuple

log = logging.getLogger(__name__)


class ColourLabel(NamedTuple):
    red: Tuple[int] = (255, 0, 0),
    green: Tuple[int] = (0, 128, 0),


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


class FrameHandler(metaclass=MetaSingleton):
    labels = ColourLabel

    def __init__(self):
        ###
        # TODO @Karim: load nn weights etc
        ###
        ...

    @staticmethod
    def preprocess_frame(frame: np.ndarray):
        ...
        ###
        # TODO @Karim: add perspective transformation
        ###

        ###
        # TODO @Karim: apply filter
        ###

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
