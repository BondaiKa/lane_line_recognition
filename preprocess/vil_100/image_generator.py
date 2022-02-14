import numpy as np
import math
from typing import Tuple, List, Optional, Type
import glob
from os.path import join, dirname
import os
from dotenv import load_dotenv
import random
import h5py
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from lane_line_recognition.utils import test_generator
from lane_line_recognition.preprocess.vil_100.utils import VIL100HDF5
import logging
from lane_line_recognition.base import AbstractFrameGenerator, AbstractFrameGeneratorCreator

log = logging.getLogger(__name__)


class VIL100FrameGenerator(AbstractFrameGenerator, Sequence):
    """Sequence of frames generator

    Usage for training NN that could process independent
    frames without context window etc
    """

    def __init__(self,
                 num_type_of_lines=4,
                 max_num_points=91,
                 max_lines_per_frame=2,
                 rescale=1 / 255.,
                 batch_size: int = 8,
                 target_shape: Tuple[int, int] = (640, 480),
                 shuffle: bool = False,
                 color_mode: str = 'grayscale',
                 files: Optional[List[str]] = None,
                 json_files: Optional[List[str]] = None):
        """
        :param max_lines_per_frame: maximum number of lines per frame
        :param max_num_points: maximum number of points un one polyline
        :param num_type_of_lines: number of possible lines on road
        :param rescale:
        :param batch_size: batch size of the dataset
        :param target_shape: final size for NN input
        :param shuffle: shuffle flag of frames sequences
        :param color_mode: `grayscale` or `rgb` color reading frame mod
        :param json_files: list of json files that contain info about a frame
        """
        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        self.rescale = rescale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.color_mode = color_mode
        self.files = files
        self.json_files = json_files
        self.files_count = len(self.files)

        if shuffle:
            temp = list(zip(self.files, self.json_files))
            random.shuffle(temp)
            self.files, self.json_files = zip(*temp)

    def __len__(self):
        return math.ceil(self.files_count / self.batch_size)

    def get_data_from_file(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get from hdf5 all polylines and their labels
        :param json_path: path of json file
        :return: polylines with labels
        """
        file = h5py.File(json_path, 'r')
        group = file.get(VIL100HDF5.GROUP_NAME)
        return group.get(VIL100HDF5.POLYLINES_DATASET_NAME), group.get(VIL100HDF5.LABELS_DATASET_NAME)

    def get_frame_from_file(self, frame_path: str) -> np.ndarray:
        frame = tf.keras.utils.load_img(frame_path,
                                        color_mode=self.color_mode,
                                        target_size=(self.target_shape[1], self.target_shape[0])
                                        )
        frame = tf.keras.preprocessing.image.img_to_array(frame)
        frame = frame * self.rescale
        frame = np.expand_dims(frame, 0)
        return frame

    def __getitem__(self, idx) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        batch_frames_path = self.files[idx * self.batch_size:
                                       (idx + 1) * self.batch_size]
        batch_json_path = self.json_files[idx * self.batch_size:
                                          (idx + 1) * self.batch_size]

        polylines_list, labels_list = self.get_data_from_file(batch_json_path[0])
        frame_list = self.get_frame_from_file(batch_frames_path[0])

        for _frame, _json in zip(batch_frames_path[1:], batch_json_path[1:]):
            polylines, labels = self.get_data_from_file(_json)
            polylines_list = np.vstack((polylines_list, polylines))
            labels_list = np.vstack((labels_list, labels))
            frame = self.get_frame_from_file(_frame)
            frame_list = np.vstack((frame_list, frame))

        return (frame_list), tuple(np.hsplit(polylines_list, self.max_lines_per_frame)) + tuple(
            np.hsplit(labels_list, self.max_lines_per_frame))


class VIL100FrameGeneratorCreator(AbstractFrameGeneratorCreator):

    def get_generator(self) -> Type[VIL100FrameGenerator]:
        return VIL100FrameGenerator


if __name__ == "__main__":
    ENV_FILE_NAME = 'vil_100.env'
    dotenv_path = join(dirname(__file__), ENV_FILE_NAME)
    load_dotenv(dotenv_path)

    AMOUNT_OF_FRAMES = 10000
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(os.getenv('NUM_TYPE_OF_LINES'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT'))
    IMAGE_PATH = os.getenv('FRAME_DATASET_PATH')
    JSON_HDF5_DATASET_PATH = os.getenv('JSON_HDF5_DATASET_PATH')

    input_shape = (CAMERA_WIDTH, CAMERA_HEIGHT, 3)
    image_glob_path = IMAGE_PATH + '/*/*.jpg'
    json_hdf5_glob_path = JSON_HDF5_DATASET_PATH + '/*/*.hdf5'

    data_gen = VIL100FrameGeneratorCreator(
        validation_split=VALIDATION_SPLIT,
        frame_glob_path=image_glob_path,
        json_hdf5_glob_path=json_hdf5_glob_path,
    )

    train_generator = data_gen.flow_from_directory(
        subset='training', shuffle=True, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES, max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS, num_type_of_lines=NUM_TYPE_OF_LINES,
        target_shape=input_shape,
    )

    validation_generator = data_gen.flow_from_directory(
        subset='validation', shuffle=True, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES, max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS, num_type_of_lines=NUM_TYPE_OF_LINES,
        target_shape=input_shape,
    )

    test_generator(train_generator, draw_line=True)
