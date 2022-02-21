from abc import ABC

from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv
from utils import TuSimpleHdf5, test_tu_simple_generator
from lane_line_recognition.base import AbstractFrameGenerator, AbstractFrameGeneratorCreator
from typing import Type, List, Tuple
import os
from os.path import join, dirname
import random
import math
import h5py
import numpy as np

Frame_ouput_type = np.ndarray
Polylines_width_ouput_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]
Polylines_height_ouput_type = Tuple[np.ndarray]


class TuSimpleFrameGenerator(AbstractFrameGenerator, Sequence):
    def __init__(self,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 batch_size: int,
                 final_shape: Tuple[int, int],
                 shuffle: bool,
                 files: List[str],
                 json_files: List[str],
                 rescale=1 / 255.,
                 color_mode: str = 'grayscale',
                 ):
        self.batch_size = batch_size
        self.final_shape = final_shape
        self.rescale = rescale
        self.color_mode = color_mode
        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points

        if shuffle:
            temp = list(zip(files, json_files))
            random.shuffle(temp)
            self.files, self.json_files = zip(*temp)

        self.files_count = len(self.files)

    def get_data_from_file(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get from hdf5 all polylines
        :param json_path: path of json file
        :return: polylines
        """
        file = h5py.File(json_path, 'r')
        group = file.get(TuSimpleHdf5.group_name)
        return group.get(TuSimpleHdf5.dataset_polylines_width), group.get(TuSimpleHdf5.dataset_polylines_height)

    def __len__(self):
        return math.ceil(self.files_count / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[Frame_ouput_type, Tuple[Polylines_width_ouput_type,
                                                                     Polylines_height_ouput_type]]:

        batch_frames_path = self.files[idx * self.batch_size:
                                       (idx + 1) * self.batch_size]
        batch_json_path = self.json_files[idx * self.batch_size:
                                          (idx + 1) * self.batch_size]

        polylines_width_output = np.empty(
            shape=(0, self.max_num_points * self.max_lines_per_frame))
        polylines_height_output = np.empty(
            shape=(0, self.max_num_points * self.max_lines_per_frame))

        frame_output = np.empty(shape=(0, self.final_shape[0], self.final_shape[1], 1))

        for _frame, _json in zip(batch_frames_path, batch_json_path):
            polylines_width, polylines_height = self.get_data_from_file(_json)

            polylines_width_output = np.vstack([polylines_width_output, polylines_width])
            polylines_height_output = np.vstack([polylines_height_output, polylines_height])

            frame = self.get_frame_from_file(_frame)
            frame_output = np.vstack((frame_output, frame))

        return frame_output, (polylines_width_output, polylines_height_output)


class TuSimpleFrameGeneratorCreator(AbstractFrameGeneratorCreator):
    def get_generator(self) -> Type[TuSimpleFrameGenerator]:
        return TuSimpleFrameGenerator


if __name__ == "__main__":
    ENV_FILE_NAME = 'tu_simple.env'
    dotenv_path = join(dirname(__file__), ENV_FILE_NAME)
    load_dotenv(dotenv_path)

    AMOUNT_OF_FRAMES = 10000
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    TU_SIMPLE_FRAME_DATASET_PATH = os.getenv('TU_SIMPLE_FRAME_DATASET_PATH')
    TU_SIMPLE_JSON_HDF5_DATASET_PATH = os.getenv('TU_SIMPLE_JSON_HDF5_DATASET_PATH')

    TU_SIMPLE_FRAME_DATASET_PATH = TU_SIMPLE_FRAME_DATASET_PATH + '/clips/*/*/*20.jpg'
    TU_SIMPLE_JSON_HDF5_DATASET_PATH = TU_SIMPLE_JSON_HDF5_DATASET_PATH + '/clips/*/*/*.hdf5'

    FINAL_SHAPE = (
        int(os.getenv('FINAL_WIDTH')),
        int(os.getenv('FINAL_HEIGHT')),
    )
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))

    tu_simple_image_generator = TuSimpleFrameGeneratorCreator(
        validation_split=0.2,
        frame_glob_path=TU_SIMPLE_FRAME_DATASET_PATH,
        json_hdf5_glob_path=TU_SIMPLE_JSON_HDF5_DATASET_PATH,
    )

    tu_simple_train_generator = tu_simple_image_generator.flow_from_directory(
        subset='training', shuffle=True, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES,
        final_shape=FINAL_SHAPE,
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
    )

    tu_simple_validation_generator = tu_simple_image_generator.flow_from_directory(
        subset='validation', shuffle=True, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES,
        final_shape=FINAL_SHAPE,
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
    )

    test_tu_simple_generator(tu_simple_train_generator, draw_line=True)
    # TODO @Karim: test generator!!!
