import logging
from abc import ABC
from typing import Union
import cv2
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from vil_100_utils import get_colour_from_one_hot_vector
from tensorflow.keras import layers, Model

log = logging.getLogger(__name__)


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
    """Draw polyline and other necessary data to frame"""

    @staticmethod
    def __build_model(polyline_output_shape: int, label_output_shape: int, input_shape=(1280, 960, 3)):
        pre_trained_model = tf.keras.applications.InceptionResNetV2(input_shape=input_shape,
                                                                    weights='imagenet',
                                                                    include_top=False)

        global_max_pool = layers.GlobalMaxPool2D()(pre_trained_model.output)
        dense_polyline = tf.keras.layers.Dense(units=512, activation='relu')(global_max_pool)
        dropout_polyine = layers.Dropout(.2)(dense_polyline)
        dense_polyline_2 = tf.keras.layers.Dense(units=512, activation='relu')(dropout_polyine)
        dropout_polyine_2 = layers.Dropout(.2)(dense_polyline_2)

        dense_label = tf.keras.layers.Dense(units=66, activation='relu')(global_max_pool)
        dropout_label = layers.Dropout(.2)(dense_label)
        dense_label_2 = tf.keras.layers.Dense(units=66, activation='relu')(dropout_label)
        dropout_label_2 = layers.Dropout(.2)(dense_label_2)

        polyline_output = layers.Dense(polyline_output_shape, name='polyline_output')(dropout_polyine_2)
        label_output = layers.Dense(label_output_shape, activation='softmax', name='label_output')(dropout_label_2)

        model = Model(pre_trained_model.input, outputs=[polyline_output, label_output])

        return model, pre_trained_model

    def __init__(self, model_weights_path: str = '', width: int = 1280,
                 height: int = 960, max_lines_per_frame: int = 8):
        """
        :param model_weights_path:  neural net weights with h5 format path
        :param width: width of camera/desired frame
        :param height: height of camera/desired frame
        :param max_lines_per_frame: maximum num of lines in a frame
        """
        model, pre_trained_model = self.__build_model(polyline_output_shape=91 * 2 * max_lines_per_frame,
                                                      label_output_shape=max_lines_per_frame * 11)
        model.load_weights(model_weights_path)
        self.model = model
        self.width = width
        self.height = height
        self.max_lines_per_frame = max_lines_per_frame

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        ###
        # TODO @Karim: apply filter
        ###
        width, height = self.width, self.height

        log.debug(f"Resizing frame to {width}x{height} resolution...")
        frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
        frame = frame / 255
        # TODO @Karim use transform later
        # presp_frame = transform_frame(frame, width, height)
        # cv2.imshow('Perspective_transform_frame', presp_frame)

        # return transform_frame(frame, width, height, reverse_flag=True)
        return frame

    def recognize(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        frame = frame.reshape(1, self.width, self.height, 3)
        return self.model.predict(frame)

    def postprocess_frame(self, polylines: np.ndarray, labels: np.ndarray) -> Tuple[List[np.ndarray]]:
        """
        Get Splitted polylines and labels values for 6 numpy arrays respectively

        :param polylines:
        :param labels:
        :return:
        """
        polylines = np.hsplit(polylines, self.max_lines_per_frame)
        polylines = self.filter_coordinates(polylines)
        colors = self.get_colour(np.hsplit(np.where(labels > 0.5, 1, 0), self.max_lines_per_frame))
        return zip(*filter(lambda poly_lab_tuple: poly_lab_tuple[1] is not None, zip(polylines, colors))) \
               or np.empty(shape=(0)), np.empty(shape=(0))

    @classmethod
    def get_colour(cls, labels: List[np.ndarray]) -> List[Tuple[int, int, int]]:
        """Get color from several line labels"""
        return list(map(lambda one_hot_v: get_colour_from_one_hot_vector(one_hot_v), labels))

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
        ###
        # TODO @Karim: try to understand `PoseArray` and further logic
        ###
        for points, color in zip(list_of_points, list_of_colors):
            frame = cv2.polylines(frame, points, True, color, thickness=10)
            cv2.imshow('Test_frame',frame)
        return frame
