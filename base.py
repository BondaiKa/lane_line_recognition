import logging
from abc import ABC
from typing import Union
import cv2
import numpy as np
from typing import NamedTuple, Tuple, List
import tensorflow as tf
from vil_100_utils import get_colour_from_one_hot_vector

log = logging.getLogger(__name__)

LABELS = {
    0: (0, 255, 0),  # green
    1: (255, 0, 0),  # red
}

NEURAL_NETWORK_MODEL_PATH = 'model/multiple-output-model.h5'


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


class FrameHandler(metaclass=MetaSingleton):
    labels = LABELS

    def __init__(self):
        ###
        # TODO @Karim: load nn weights etc
        ###
        ...

    @staticmethod
    def preprocess_frame(frame: np.ndarray, width: int, height: int):
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
        presp_frame = transform_frame(frame, width, height)
        cv2.imshow('Perspective_transform_frame', presp_frame)
        reversed_frame = transform_frame(frame, width, height, reverse_flag=True)
        return reversed_frame

    @staticmethod
    def recognize(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ###
        # TODO @Karim: add process image by NN
        ###
        model = tf.keras.models.load_model(NEURAL_NETWORK_MODEL_PATH)
        frame = frame.reshape(1, 1280, 960, 3)
        return model.predict(frame)

    @classmethod
    def postprocess_frame(cls, polylines: np.ndarray, labels: np.ndarray) -> Tuple[List[np.ndarray]]:
        """
        Get Splitted polylines and labels values for 6 numpy arrays respectively

        :param polylines:
        :param labels:
        :return:
        """
        polylines = np.hsplit(polylines, 6)
        polylines = cls.filter_coordinates(polylines)
        colors = cls.get_colour(np.hsplit(np.where(labels > 0.5, 1, 0), 6))
        return zip(*filter(lambda poly_lab_tuple: poly_lab_tuple[1] is not None, zip(polylines, colors)))

    @classmethod
    def get_colour(cls, labels: List[np.ndarray]) -> List[Tuple[int, int, int]]:
        """Get color from several line labels"""
        return list(map(lambda one_hot_v: get_colour_from_one_hot_vector(one_hot_v), labels))

    @classmethod
    def filter_coordination_for_resolution(cls, polyline: np.ndarray) -> np.ndarray:
        valid = ((polyline[:, 0] > 0) & (polyline[:, 1] > 0)
                 & (polyline[:, 0] < 1280) & (polyline[:, 1] < 960))
        return polyline[valid]

    @classmethod
    def filter_coordinates(cls, list_of_polylines: List[np.ndarray]) -> np.ndarray:
        """Remove empty points and coordinates x or y, that is less than 0"""
        list_of_polylines = list(map(lambda x: x.reshape(-1, 2), list_of_polylines))
        return list(map(lambda polyline: cls.filter_coordination_for_resolution(polyline),
                        list_of_polylines))

    @classmethod
    def draw_popylines(cls, frame: np.ndarray, list_of_points: List[np.ndarray],
                       list_of_colors: List[np.ndarray]) -> np.ndarray:
        """
        draw polylines and labels to a frame
        :param frame: input frame
        :param list_of_points: list of polylines
        :param list_of_labels: list of label that corresponds to list of polylines
        :return:
        """
        ###
        # TODO @Karim: try to understand `PoseArray` and further logic
        ###
        for points, color in zip(list_of_points, list_of_colors):
            frame = cv2.polylines(frame, points, True, color, thickness=2)
        return frame
