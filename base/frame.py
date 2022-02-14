from .base import MetaSingleton
from lane_line_recognition.preprocess.vil_100 import get_colour_from_one_hot_vector
from typing import Tuple, List
import numpy as np
import cv2
import tensorflow as tf

import logging

log = logging.getLogger(__name__)


class FrameHandler(metaclass=MetaSingleton):
    """Draw polyline and other necessary data to frame"""

    def __init__(self, model_path: str, width: int,
                 height: int,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 num_type_of_lines: int,
                 neural_net_width: int,
                 neural_net_height: int,
                 rescale_polyline_coef: float
                 ):
        """
        :param num_type_of_lines: max lane line type (dotted, solid etc)
        :param model_weights_path:  neural net weights with h5 format path
        :param width: width of camera/desired frame
        :param height: height of camera/desired frame
        :param max_lines_per_frame: maximum num of lines in a frame
        :param max_num_points: maximum number of points(x,y) per polylines
        """
        model = tf.keras.models.load_model(model_path)
        self.model = model
        self.width = width
        self.height = height
        self.max_lines_per_frame = max_lines_per_frame
        self.neural_net_width = neural_net_width
        self.neural_net_height = neural_net_height
        self.rescale_polyline_coef = rescale_polyline_coef

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        ###
        # TODO @Karim: apply filter
        ###
        width, height = self.width, self.height

        log.debug(f"Resizing frame to {self.neural_net_width}x{self.neural_net_height} resolution...")
        frame = cv2.resize(frame, dsize=(self.neural_net_width, self.neural_net_height), interpolation=cv2.INTER_AREA)
        frame = frame / 255
        # TODO @Karim use transform later
        # presp_frame = transform_frame(frame, width, height)
        # cv2.imshow('Perspective_transform_frame', presp_frame)

        # return transform_frame(frame, width, height, reverse_flag=True)
        return frame

    def recognize(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        frame = tf.image.rgb_to_grayscale(frame).numpy()
        frame = frame.reshape(1, self.neural_net_width, self.neural_net_height, 1)
        return self.model.predict(frame)

    def postprocess_frame(self, polylines: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[List[np.ndarray]]:
        """
        Get Splitted polylines and labels values (number of line)-n numpy arrays respectively

        :param polylines:
        :param labels:
        :return:
        """
        polylines = self.filter_coordinates(polylines)
        polylines = [polyline / self.rescale_polyline_coef for polyline in polylines]
        colors = list(map(lambda label: get_colour_from_one_hot_vector(np.where(label > 0.5, 1, 0)), labels))
        res = tuple(zip(*filter(lambda poly_lab_tuple: poly_lab_tuple[1] is not None, zip(polylines, colors))))
        return res if res else (list(), list())

    def filter_coordination_for_resolution(self, polyline: np.ndarray) -> np.ndarray:
        valid = ((polyline[:, 0] > 0) & (polyline[:, 1] > 0)
                 & (polyline[:, 0] < self.width) & (polyline[:, 1] < self.height))
        return polyline[valid]

    def filter_coordinates(self, list_of_polylines: List[np.ndarray]) -> np.ndarray:
        """Remove empty points and coordinates x or y, that is less than 0"""
        list_of_polylines = list(map(lambda x: x.reshape(-1, 2), list_of_polylines))
        return list(map(lambda polyline: self.filter_coordination_for_resolution(polyline),
                        list_of_polylines))

    @classmethod
    def draw_popylines(cls, frame: np.ndarray, list_of_points: List[np.ndarray],
                       list_of_colors: List[np.ndarray]) -> np.ndarray:
        """
        draw polylines and labels to a frame
        :param list_of_colors: list of color for each polyline
        :param frame: input frame
        :param list_of_points: list of polylines
        :param list_of_labels: list of label that corresponds to list of polylines
        :return:
        """
        for points, color in zip(list_of_points, list_of_colors):
            frame = cv2.polylines(frame, np.int32(points).reshape((-1, 1, 2)), 1, color, thickness=5)
        return frame
