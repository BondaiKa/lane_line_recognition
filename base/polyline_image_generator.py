from lane_line_recognition.base import AbstractFrameGenerator, AbstractFrameGeneratorCreator
from lane_line_recognition.base.utils import LaneLineRecognitionHDF5, test_generator
from tensorflow.keras.utils import Sequence
from typing import Tuple, List, Optional, Type
import h5py
import numpy as np
import math
import random
import os
from dotenv import load_dotenv

import logging

log = logging.getLogger(__name__)


class PolylineFrameGenerator(AbstractFrameGenerator, Sequence):
    """Sequence of frames generator

    Usage for training NN that could process independent
    frames without context window etc
    """

    def get_data_from_file(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get from hdf5 all line polygonal chains
        :param json_path: path of json file
        :return: polylines
        """
        file = h5py.File(json_path, 'r')
        group = file.get(LaneLineRecognitionHDF5.group_name)
        return group.get(LaneLineRecognitionHDF5.polyline_widths_dataset_name), \
               group.get(LaneLineRecognitionHDF5.polyline_heights_dataset_name)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        batch_frames_path = self.files[idx * self.batch_size:
                                       (idx + 1) * self.batch_size]
        batch_json_path = self.json_files[idx * self.batch_size:
                                          (idx + 1) * self.batch_size]

        polyline_width_output = np.empty(shape=(0, self.max_lines_per_frame * self.max_num_points))
        polyline_height_output = np.empty(shape=(0, self.max_lines_per_frame * self.max_num_points))

        frame_output = np.empty(shape=(0, self.final_shape[0], self.final_shape[1], 1))

        for _frame, _json in zip(batch_frames_path, batch_json_path):
            polylines = self.get_data_from_file(_json)
            polyline_widths, polyline_heights = polylines[0], polylines[1]

            polyline_width_output = np.vstack((polyline_width_output, polyline_widths))
            polyline_height_output = np.vstack((polyline_height_output, polyline_heights))

            frame = self.get_frame_from_file(_frame)
            frame_output = np.vstack((frame_output, frame))

        return frame_output, (polyline_width_output, polyline_height_output)


class LaneLineFrameGeneratorCreator(AbstractFrameGeneratorCreator):

    def get_generator(self) -> Type[Sequence]:
        return PolylineFrameGenerator


if __name__ == "__main__":
    ENV_FILE_NAME = '.env'
    dotenv_path = os.path.join(os.path.dirname(__file__), ENV_FILE_NAME)
    load_dotenv()

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

    data_gen = LaneLineFrameGeneratorCreator(
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

    test_generator(
        train_generator,
        draw_line=True,
        input_shape=input_shape,
        max_lines_per_frame=MAX_LINES_PER_FRAME
    )
