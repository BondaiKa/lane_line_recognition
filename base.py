import logging
from abc import ABC
from typing import Union
import cv2
import numpy as np
from typing import Tuple, List
from lane_line_recognition.vil_100.utils import get_colour_from_one_hot_vector
import tensorflow as tf

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
        polylines = [ polyline / self.rescale_polyline_coef for polyline in polylines]
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
