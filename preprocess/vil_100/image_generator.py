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
from lane_line_recognition.preprocess.vil_100.utils import VIL100HDF5, test_vil100_generator
import logging
from lane_line_recognition.base import AbstractFrameGenerator, AbstractFrameGeneratorCreator

log = logging.getLogger(__name__)


class VIL100FrameGenerator(AbstractFrameGenerator, Sequence):
    """Sequence of frames generator

    Usage for training NN that could process independent
    frames without context window etc
    """

    def __init__(self,
                 num_type_of_lines,

                 max_lines_per_frame,
                 max_num_points,
                 batch_size: int,
                 final_shape: Tuple[int, int],
                 shuffle: bool,
                 files: Optional[List[str]],
                 json_files: Optional[List[str]],
                 rescale=1 / 255.,
                 color_mode: str = 'grayscale',
                 ):
        """
        :param max_lines_per_frame: maximum number of lines per frame
        :param max_num_points: maximum number of points un one polyline
        :param num_type_of_lines: number of possible lines on road
        :param rescale:
        :param batch_size: batch size of the dataset
        :param final_shape: final size for NN input
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
        self.final_shape = final_shape
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

    def get_data_from_file(self, json_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Get from hdf5 all polylines and their labels
        :param json_path: path of json file
        :return: polylines with labels
        """
        file = h5py.File(json_path, 'r')
        group = file.get(VIL100HDF5.GROUP_NAME)
        return (group.get(VIL100HDF5.POLYLINE_WIDTHS_DATASET_NAME),
                group.get(VIL100HDF5.POLYLINE_HEIGHTS_DATASET_NAME)
                ), group.get(VIL100HDF5.LABELS_DATASET_NAME)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        batch_frames_path = self.files[idx * self.batch_size:
                                       (idx + 1) * self.batch_size]
        batch_json_path = self.json_files[idx * self.batch_size:
                                          (idx + 1) * self.batch_size]

        polyline_width_output = np.empty(shape=(0, self.max_lines_per_frame * self.max_num_points))
        polyline_height_output = np.empty(shape=(0, self.max_lines_per_frame * self.max_num_points))
        labels_output = np.empty(shape=(0, self.num_type_of_lines * self.max_lines_per_frame))
        frame_output = np.empty(shape=(0, self.final_shape[0], self.final_shape[1], 1))

        for _frame, _json in zip(batch_frames_path, batch_json_path):
            polylines, labels = self.get_data_from_file(_json)
            polyline_widths, polyline_heights = polylines[0], polylines[1]

            polyline_width_output = np.vstack((polyline_width_output, polyline_widths))
            polyline_height_output = np.vstack((polyline_height_output, polyline_heights))
            labels_output = np.vstack((labels_output, labels))
            frame = self.get_frame_from_file(_frame)
            frame_output = np.vstack((frame_output, frame))

        return frame_output, (polyline_width_output, polyline_height_output)


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
    FINAL_WIDTH = int(os.getenv('FINAL_WIDTH'))
    FINAL_HEIGHT = int(os.getenv('FINAL_HEIGHT'))
    IMAGE_PATH = os.getenv('FRAME_DATASET_PATH')
    JSON_HDF5_DATASET_PATH = os.getenv('JSON_HDF5_DATASET_PATH')

    input_shape = (FINAL_WIDTH, FINAL_HEIGHT, 1)
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
        final_shape=input_shape,
    )

    validation_generator = data_gen.flow_from_directory(
        subset='validation', shuffle=True, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES, max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS, num_type_of_lines=NUM_TYPE_OF_LINES,
        final_shape=input_shape,
    )

    test_vil100_generator(
        train_generator,
        draw_line=True,
        input_shape=input_shape,
        max_lines_per_frame=MAX_LINES_PER_FRAME
    )
