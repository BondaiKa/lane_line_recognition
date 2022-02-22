# Tu Simple Json label converter
import h5py
import json
import numpy as np
from dotenv import load_dotenv
import os
from typing import Tuple
import logging
import cv2
from os.path import join, dirname
from utils import TuSimpleJson
from lane_line_recognition.base import AbstractConverter, LaneLineRecognitionHDF5
from pathlib import Path

log = logging.getLogger(__name__)


class TuSimpleJsonConverter(AbstractConverter):
    def __init__(self,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 json_file_path: str,
                 final_shape_to_convert: Tuple[int, int],
                 frame_dataset_path: str,
                 final_binary_json_path: str,
                 ):
        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.json_file = json_file_path
        self.frame_dataset_path = frame_dataset_path
        self.final_shape_to_convert = final_shape_to_convert
        self.TU_SIMPLE_EXPECTED_SHAPE = (1280, 720)
        self.final_binary_json_path = final_binary_json_path

        log.debug(f'TuSimpleJsonConverter params: {locals()}')

    def _verify_frame_shape(self, frame_path: np.ndarray) -> Tuple[int, int]:
        """Check that each frame has same expected shape"""
        frame = cv2.imread(frame_path)
        if not frame:
            log.warning(f'Frame not found. Frame path: {frame_path}.')
            raise FileNotFoundError

        height, width = frame.shape[0], frame.shape[1]
        if self.TU_SIMPLE_EXPECTED_SHAPE[0] != width or \
                self.TU_SIMPLE_EXPECTED_SHAPE[1] != height:
            log.warning('Shape of frame and expected is not equal! '
                        f'Real shape:{width}x{height}')
            raise ValueError('Shape of frame and expected is not equal! '
                             f'Real shape:{width}x{height}')
        return width, height

    def __transform_widths(self, original_width: int, polyline_widths: np.ndarray):
        """Scale widths data"""
        polyline_widths = np.where(
            polyline_widths > 0,
            polyline_widths / original_width,
            polyline_widths
        )
        polyline_widths = np.where(polyline_widths == -2, -1, polyline_widths)
        polyline_widths = np.apply_along_axis(
            func1d=np.pad,
            axis=1,
            arr=polyline_widths,
            pad_width=(0, self.max_num_points - polyline_widths.shape[1]),
            mode='constant',
            constant_values=(-1,)
        )

        #TODO @Karim: refactor if condition
        if self.max_lines_per_frame - polyline_widths.shape[0]:
            polyline_widths = polyline_widths[:self.max_lines_per_frame]

        empty_polylines = np.full(shape=((self.max_lines_per_frame - polyline_widths.shape[0]) * self.max_num_points),
                                  fill_value=-1)
        return np.hstack([polyline_widths.flatten(), empty_polylines])

    def __transform_heights(self, original_height: int, polyline_heights: np.ndarray) -> np.ndarray:
        """Scale heights data"""
        polyline_heights = np.where(
            polyline_heights > 0,
            polyline_heights / original_height,
            polyline_heights
        )
        polyline_heights = np.pad(polyline_heights, pad_width=(0, self.max_num_points - polyline_heights.shape[0]),
                                  mode='constant',
                                  constant_values=(-1,))
        return np.tile(polyline_heights, self.max_lines_per_frame)

    def transform_polylines(self, original_width: int, original_height: int,
                            polyline_widths: np.ndarray, polyline_heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale polygonal chains: divide and standardize dataset for NN input
        :param original_width: initial frame width
        :param original_height:initial frame height
        :param polyline_widths: polygonal widths
        :param polyline_heights: polygonal heights
        :return: scaled and prepared polygonal widths and heights
        """
        # TODO @Karim: check adding -1 at the start of each array!!!
        polyline_widths = self.__transform_widths(original_width=original_width, polyline_widths=polyline_widths)
        polyline_heights = self.__transform_heights(original_height=original_height, polyline_heights=polyline_heights)
        return polyline_widths, polyline_heights

    def get_polylines_from_json_line(self, line: str) -> Tuple[str, np.ndarray, np.ndarray]:
        """Retrieve polygonal chains from json file"""
        json_line = json.loads(line)
        frame_path = json_line.pop(TuSimpleJson.frame_path)
        frame_full_path = self.frame_dataset_path + '/' + frame_path
        width, height = self._verify_frame_shape(frame_full_path)
        lane_widths, lane_heights = self.transform_polylines(
            original_width=width,
            original_height=height,
            polyline_widths=np.array(json_line[TuSimpleJson.lane_widths]),
            polyline_heights=np.array(json_line[TuSimpleJson.lane_heights])
        )
        return frame_path, lane_widths, lane_heights

    def get_data_from_file(self, json_path: str):
        """Read json line by line"""
        # TODO @Karim: investigate and decide to add labels
        with open(json_path, 'r') as f:
            for json_frame_data_line in f.readlines():
                yield self.get_polylines_from_json_line(json_frame_data_line)

    def exec(self):
        for frame_path, polylines_width, polylines_height in self.get_data_from_file(self.json_file):
            full_path = self.final_binary_json_path + '/' + '/'.join(frame_path.split('/')[:-1])
            Path(full_path).mkdir(parents=True, exist_ok=True)

            with h5py.File(f"{self.final_binary_json_path}/{frame_path}.hdf5", "w") as f:
                grp = f.create_group(LaneLineRecognitionHDF5.group_name)
                grp.create_dataset(LaneLineRecognitionHDF5.polyline_widths_dataset_name, data=polylines_width,
                                   dtype='float32')
                grp.create_dataset(LaneLineRecognitionHDF5.polyline_heights_dataset_name, data=polylines_height,
                                   dtype='float32')


if __name__ == '__main__':
    ENV_FILE_NAME = 'tu_simple.env'
    dotenv_path = join(dirname(__file__), ENV_FILE_NAME)
    load_dotenv(dotenv_path)

    FINAL_WIDTH = int(os.getenv('FINAL_WIDTH'))
    FINAL_HEIGHT = int(os.getenv('FINAL_HEIGHT'))
    TU_SIMPLE_JSON_PATH = os.getenv('TU_SIMPLE_JSON_PATH')
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    TU_SIMPLE_JSON_HDF5_DATASET_PATH = os.getenv('TU_SIMPLE_JSON_HDF5_DATASET_PATH')
    TU_SIMPLE_FRAME_DATASET_PATH = os.getenv('TU_SIMPLE_FRAME_DATASET_PATH')

    tu_simple_converter = TuSimpleJsonConverter(
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        json_file_path=TU_SIMPLE_JSON_PATH,
        final_shape_to_convert=(FINAL_WIDTH, FINAL_HEIGHT),
        frame_dataset_path=TU_SIMPLE_FRAME_DATASET_PATH,
        final_binary_json_path=TU_SIMPLE_JSON_HDF5_DATASET_PATH,
    )
    tu_simple_converter.exec()
    log.info('Done...')
