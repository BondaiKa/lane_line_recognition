# Tu Simple Json label converter
import h5py
import json
import numpy as np
from dotenv import load_dotenv
import os
from typing import Tuple
import logging
import cv2
from utils import TuSimpleJson, TuSimpleHdf5
from pathlib import Path

log = logging.getLogger(__name__)


class TuSimpleJsonConverter:
    # TODO @Karim: check frame's resolution. Is it `1280×720`?
    # TODO @Karim: investigate and decide to add labels
    def __init__(self,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 json_file_path: str,
                 final_shape_to_convert: Tuple[int, int],
                 frame_dataset_path: str,
                 rescale_polyline_coef: float,
                 final_binary_json_path: str,
                 ):
        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.json_file = json_file_path
        self.frame_dataset_path = frame_dataset_path
        self.final_shape_to_convert = final_shape_to_convert
        self.rescale_polyline_coef = rescale_polyline_coef
        self.TU_SIMPLE_EXPECTED_SHAPE = (1280, 720)
        self.final_binary_json_path = final_binary_json_path

    def _verify_frame_shape(self, frame_path: np.ndarray) -> None:
        frame = cv2.imread(frame_path)
        height, width = frame.shape[0], frame.shape[1]
        if self.TU_SIMPLE_EXPECTED_SHAPE[0] != width or \
                self.TU_SIMPLE_EXPECTED_SHAPE[1] != height:
            log.warning('Shape of frame and expected is not equal! '
                        f'Real shape:{width}x{height}')
            raise ValueError('Shape of frame and expected is not equal! '
                             f'Real shape:{width}x{height}')

    def get_polylines_from_json_line(self, line: str) ->Tuple[np.ndarray, str]:
        json_line = json.loads(line)
        frame_path = json_line.pop(TuSimpleJson.frame_path)
        frame_full_path = self.frame_dataset_path + frame_path
        self._verify_frame_shape(frame_full_path)

        lanes = np.array(json_line[TuSimpleJson.lanes])
        lanes = np.where(lanes > 0, lanes / self.TU_SIMPLE_EXPECTED_SHAPE[0] * self.final_shape_to_convert[0], lanes)
        lanes = np.where(lanes == -2, -1, lanes)
        return lanes, frame_path

    def exec(self):
        with open(self.json_file, 'r') as f:
            for json_frame_data_line in f.readlines():
                polylines_width, frame_path = self.get_polylines_from_json_line(json_frame_data_line)

                Path(self.final_binary_json_path).mkdir(parents=True, exist_ok=True)

                with h5py.File(f"{self.final_binary_json_path}/{frame_path}.hdf5", "w") as f:
                    grp = f.create_group(TuSimpleHdf5.group_name)
                    grp.create_dataset(TuSimpleHdf5.dataset_name, data=polylines_width, dtype='float32')


if __name__ == '__main__':
    load_dotenv()
    NEURAL_NETWORK_WIDTH = int(os.getenv('NEURAL_NETWORK_WIDTH'))
    NEURAL_NETWORK_HEIGHT = int(os.getenv('NEURAL_NETWORK_HEIGHT'))
    TU_SIMPLE_JSON_PATH = os.getenv('TU_SIMPLE_JSON_PATH')
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    TU_SIMPLE_JSON_HDF5_DATASET_PATH = os.getenv('TU_SIMPLE_JSON_HDF5_DATASET_PATH')
    TU_SIMPLE_FRAME_DATASET_PATH = os.getenv('TU_SIMPLE_FRAME_DATASET_PATH')

    tu_simple_converter = TuSimpleJsonConverter(
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        json_file_path=TU_SIMPLE_JSON_PATH,
        final_shape_to_convert=(NEURAL_NETWORK_WIDTH, NEURAL_NETWORK_HEIGHT),
        frame_dataset_path=TU_SIMPLE_FRAME_DATASET_PATH,
        final_binary_json_path=TU_SIMPLE_JSON_HDF5_DATASET_PATH,

    )
    tu_simple_converter.exec()
    log.info('Done...')
