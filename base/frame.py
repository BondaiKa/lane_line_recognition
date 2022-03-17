from lane_line_recognition.base import MetaSingleton
from lane_line_recognition.preprocess.vil_100 import get_colour_from_one_hot_vector
from typing import Tuple, List
import numpy as np
import cv2
import tensorflow as tf
import os
import logging

log = logging.getLogger(__name__)


class FrameHandler(metaclass=MetaSingleton):
    """Draw polyline and other necessary data to frame"""

    def __init__(self,
                 polyline_model_path: str,
                 label_model_path: str,
                 width: int,
                 height: int,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 num_type_of_lines: int,
                 neural_net_width: int,
                 neural_net_height: int,
                 ):
        """
        :param num_type_of_lines: max lane line type (dotted, solid etc)
        :param polyline_model_path:  neural net for polyline recognition with h5 format path
        :param label_model_path:  neural net for label recognition with h5 format path
        :param width: width of camera/desired frame
        :param height: height of camera/desired frame
        :param max_lines_per_frame: maximum num of lines in a frame
        :param max_num_points: maximum number of points(x,y) per polylines
        """
        polyline_model = tf.keras.models.load_model(polyline_model_path)
        label_model = tf.keras.models.load_model(label_model_path)
        self.polyline_model = polyline_model
        self.label_model = label_model
        self.width = width
        self.height = height
        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        self.neural_net_width = neural_net_width
        self.neural_net_height = neural_net_height

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        ###
        # TODO @Karim: apply filter
        ###
        width, height = self.width, self.height
        _frame = np.copy(frame)
        log.debug(f"Resizing frame to {self.neural_net_width}x{self.neural_net_height} resolution...")
        _frame = cv2.resize(_frame, dsize=(self.neural_net_width, self.neural_net_height), interpolation=cv2.INTER_AREA)
        _frame = _frame / 255
        # TODO @Karim use transform later
        # presp_frame = transform_frame(frame, width, height)
        # cv2.imshow('Perspective_transform_frame', presp_frame)

        # return transform_frame(frame, width, height, reverse_flag=True)
        return _frame

    def recognize(self, frame: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        frame = tf.image.rgb_to_grayscale(frame).numpy()
        frame = frame.reshape(1, self.neural_net_width, self.neural_net_height, 1)
        polyline_widths, polyline_height = self.polyline_model.predict(frame)
        label_1, label_2 = self.label_model.predict(frame)
        return (polyline_widths, polyline_height), (label_1, label_2)

    def postprocess_frame(self, polylines: Tuple[np.ndarray, np.ndarray], labels: Tuple[np.ndarray]) \
            -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get Separated polylines and labels values (number of line)-n numpy arrays respectively

        :param polylines:
        :param labels:
        :return:
        """
        polyline_widths, polyline_height = polylines
        polyline_widths *= self.neural_net_width
        polyline_height *= self.neural_net_height
        polylines = self._concat_polyline(polyline_width=polyline_widths.reshape(-1, self.max_num_points),polyline_height=polyline_height)
        # TODO @Karim: we can't just reshape because we have to recognize which label corresponds coordinate
        polylines = self.filter_coordinates(np.split(polylines, self.max_lines_per_frame))
        colors = list(map(lambda label: get_colour_from_one_hot_vector(np.where(label.flatten() > 0.5, 1, 0)), labels))
        res = tuple(zip(*filter(lambda poly_lab_tuple: poly_lab_tuple[1] is not None, zip(polylines, colors))))
        return res if res else (list(), list())

    @staticmethod
    def _concat_polyline(polyline_width: np.ndarray, polyline_height: np.ndarray) -> np.ndarray:
        return np.concatenate((polyline_width.reshape(-1, 1), polyline_height.T), axis=1)

    def filter_coordination_for_resolution(self, polyline: np.ndarray) -> np.ndarray:
        valid = ((polyline[:, 0] > 0) & (polyline[:, 1] > 0)
                 & (polyline[:, 0] < self.width) & (polyline[:, 1] < self.height))
        return polyline[valid]

    def filter_coordinates(self, list_of_polylines: List[np.ndarray]) -> np.ndarray:
        """Remove empty points and coordinates x or y, that is less than 0"""
        list_of_polylines = list(map(lambda x: x.reshape(-1, 2), list_of_polylines))
        return list(map(lambda polyline: self.filter_coordination_for_resolution(polyline),
                        list_of_polylines))

    def rescale_polylines(self, polylines: List[np.ndarray]) -> List[np.ndarray]:
        # TODO: @Karim rescale polylines to initial resolution
        return polylines

    @classmethod
    def draw_popylines(cls, frame: np.ndarray, list_of_points: List[np.ndarray],
                       list_of_colors: List[np.ndarray]) -> np.ndarray:
        """
        draw polygonal chains and labels to a frame
        :param list_of_colors: list of color for each polyline
        :param frame: input frame
        :param list_of_points: list of polylines
        :param list_of_labels: list of label that corresponds to list of polylines
        :return:
        """
        for points, color in zip(list_of_points, list_of_colors):
            frame = cv2.polylines(frame, np.int32(points).reshape((-1, 1, 2)), 1, color, thickness=3)
        return frame


if __name__ == '__main__':
    from dotenv import load_dotenv
    log.info("Start working...")
    load_dotenv()

    FRAME_PATH = os.getenv('FRAME_PATH')
    POLYLINE_NEURAL_NETWORK_MODEL_PATH = os.getenv('POLYLINE_NEURAL_NETWORK_MODEL_PATH')
    LABEL_NEURAL_NETWORK_MODEL_PATH = os.getenv('LABEL_NEURAL_NETWORK_MODEL_PATH')
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT'))
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(os.getenv('NUM_TYPE_OF_LINES'))
    NEURAL_NETWORK_WIDTH = int(os.getenv('NEURAL_NETWORK_WIDTH'))
    NEURAL_NETWORK_HEIGHT = int(os.getenv('NEURAL_NETWORK_HEIGHT'))

    frame_handler = FrameHandler(
        polyline_model_path=POLYLINE_NEURAL_NETWORK_MODEL_PATH,
        label_model_path=LABEL_NEURAL_NETWORK_MODEL_PATH,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        num_type_of_lines=NUM_TYPE_OF_LINES,
        neural_net_width=NEURAL_NETWORK_WIDTH,
        neural_net_height=NEURAL_NETWORK_WIDTH,
    )

    initial_frame = cv2.imread(FRAME_PATH)
    cv2.imshow(f'Original frame', initial_frame)
    frame = frame_handler.preprocess_frame(initial_frame)
    polylines, labels = frame_handler.recognize(frame)
    polylines, colors = frame_handler.postprocess_frame(polylines=polylines, labels=labels)
    result = frame_handler.draw_popylines(frame=frame, list_of_points=polylines, list_of_colors=colors)

    # rescaled_polylines = frame_handler.rescale_polylines(polylines)
    # labeled_intial_frame = frame_handler.draw_popylines(frame=initial_frame, list_of_points=rescaled_polylines,
    #                                                     list_of_colors=colors)
    cv2.imshow(f'Final frame', result)
    cv2.waitKey(1)
