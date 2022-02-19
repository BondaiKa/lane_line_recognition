import h5py
from typing import List, Dict, Tuple
import json
from os.path import join, dirname
from lane_line_recognition.preprocess.vil_100.utils import Vil100Json, LANE_ID_FULL_LIST, LineType, get_valid_attribute
import numpy as np
from lane_line_recognition.utils import one_hot_list_encoder
import glob
import os
from pathlib import Path
from lane_line_recognition.preprocess.vil_100.utils import VIL100HDF5
from fix_json_files import JsonReviewer
import cv2
from typing import Union
from lane_line_recognition.base import AbstractConverter

from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)

Lane_type = List[Dict[str, int]]

polyline_width_type = np.ndarray
polyline_height_type = np.ndarray
labels_type = np.ndarray


class VILJsonConverter(AbstractConverter):
    def __init__(self,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 num_type_of_lines: int,
                 json_glob_path: str,
                 final_shape: Tuple[int, int],
                 frame_dataset_path: str,
                 ):

        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        self.json_files = sorted(glob.glob(json_glob_path))
        self.files_count = len(self.json_files)
        self.frame_dataset_path = frame_dataset_path
        self.final_shape = final_shape

        message = f"VILJsonConverter params:\n {locals()}"
        log.debug(message)
        print(message)

    @staticmethod
    def __rescale_coordinates(row: np.ndarray, initial_width: int, initial_height: int) -> np.ndarray:
        return row[0] / initial_width, row[1] / initial_height

    def __rescale_polylines(self, polylines: np.ndarray, initial_width: int, initial_height: int) -> np.ndarray:
        """Rescale coordinates due to new frame resolution"""
        return np.apply_along_axis(self.__rescale_coordinates, axis=1, arr=polylines,
                                   initial_width=initial_width,
                                   initial_height=initial_height
                                   )

    def __get_polyline_with_label(self, lane: dict, initial_width: int, initial_height: int) \
            -> Tuple[Tuple[polyline_width_type, polyline_height_type], labels_type]:
        """Get array from points list"""
        points = np.array(
            lane[Vil100Json.POINTS])  # todo: fix
        # widths, height = np.split(points, 2, axis=1)
        points = self.__rescale_polylines(points, initial_width=initial_width, initial_height=initial_height).flatten()
        points = np.pad(points, pad_width=(0, self.max_num_points * 2 - points.shape[0]),
                        mode='constant', constant_values=(-1,))
        points = tuple(np.split(points.reshape(-1, 2), 2, axis=1))
        polyline_widths, polyline_heights = points[0].flatten(), points[1].flatten()
        # TODO @Karim: remember below `label.get(label)` is index 1,2,3,4
        label = get_valid_attribute(lane.get(Vil100Json.ATTRIBUTE, LineType.NO_LINE))
        labels = one_hot_list_encoder(label, self.num_type_of_lines)
        return (polyline_widths, polyline_heights), labels

    @staticmethod
    def _get_frame_shape(frame_path: np.ndarray) -> Tuple[int, int]:
        """Check that each frame has same expected shape"""
        frame = cv2.imread(frame_path)
        height, width = frame.shape[0], frame.shape[1]
        return width, height

    def get_data_from_file(self, json_path: str) -> Tuple[Tuple[polyline_width_type, polyline_height_type], np.ndarray]:
        """
        Retrieve from json file polylines and labels and format to nn input

        :param json_path: json file path
        :return: frame and tuple of labels
        """
        with open(json_path) as f:
            json_file: Dict[str, Union[int, dict]] = json.load(f)

        image_path = json_file[Vil100Json.INFO][Vil100Json.IMAGE_PATH]
        full_frame_path = self.frame_dataset_path + '/' + image_path
        width, height = self._get_frame_shape(frame_path=full_frame_path)

        json_file = JsonReviewer.fix_json_file(
            json_file=json_file,
            frame_real_height=height,
            frame_real_width=width,
            frame_path=image_path,
        )

        lanes: Lane_type = json_file[Vil100Json.ANNOTATIONS][Vil100Json.LANE]
        lanes = sorted(lanes, key=lambda lane: lane[Vil100Json.LANE_ID])

        polyline_widths_output, polyline_heights_output = np.empty(shape=(0, self.max_num_points)), np.empty(
            shape=(0, self.max_num_points))
        labels = np.empty(shape=(0, self.num_type_of_lines))
        # TODO @Karim: check another params in json files like "occlusion"
        exist_lane = [x[Vil100Json.LANE_ID] for x in lanes]
        missed_lane = LANE_ID_FULL_LIST - set(exist_lane)

        for lane_id in range(1, self.max_lines_per_frame + 1):
            if lane_id in missed_lane:
                points: Tuple[polyline_width_type, polyline_height_type] = (
                    np.full(shape=self.max_num_points, fill_value=-1),
                    np.full(shape=self.max_num_points, fill_value=-1)
                )
                label = one_hot_list_encoder(LineType.NO_LINE, self.num_type_of_lines)
            else:
                points, label = self.__get_polyline_with_label(
                    lane=lanes[exist_lane.index(lane_id)],
                    initial_width=width,
                    initial_height=height
                )
            polyline_widths, polyline_heights = points[0], points[1]

            if lane_id % 2 == 0:
                polyline_widths_output = np.append(polyline_widths_output, polyline_widths)
                polyline_heights_output = np.append(polyline_heights_output, polyline_heights)
                labels = np.append(labels, label)
            else:
                polyline_widths_output = np.insert(polyline_widths_output, 0, polyline_widths)
                polyline_heights_output = np.insert(polyline_heights_output, 0, polyline_heights)
                labels = np.insert(labels, 0, label)

        return (polyline_widths_output, polyline_heights_output), labels

    def exec(self) -> None:
        """Convert and save json files to new hdf5 files"""
        for json_file_path in self.json_files:
            polylines, labels = self.get_data_from_file(json_file_path)
            polyline_widths, polyline_heights = polylines[0], polylines[1]
            full_path_list = json_file_path.split('/')
            full_path_list[-3] = VIL100HDF5.ROOT_FOLDER
            root_path = full_path_list[:-1]
            frame_name = full_path_list[-1]

            Path(f"{'/'.join(root_path)}").mkdir(parents=True, exist_ok=True)

            with h5py.File(f"{'/'.join(root_path)}/{frame_name}.hdf5", "w") as f:
                grp = f.create_group(VIL100HDF5.GROUP_NAME)
                grp.create_dataset(VIL100HDF5.POLYLINE_WIDTHS_DATASET_NAME, data=polyline_widths, dtype='float32')
                grp.create_dataset(VIL100HDF5.POLYLINE_HEIGHTS_DATASET_NAME, data=polyline_heights, dtype='float32')
                grp.create_dataset(VIL100HDF5.LABELS_DATASET_NAME, data=labels, dtype='float32')


if __name__ == '__main__':
    ENV_FILE_NAME = 'vil_100.env'
    dotenv_path = join(dirname(__file__), ENV_FILE_NAME)
    load_dotenv(dotenv_path)

    FINAL_WIDTH = int(os.getenv('FINAL_WIDTH'))
    FINAL_HEIGHT = int(os.getenv('FINAL_HEIGHT'))
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(os.getenv('NUM_TYPE_OF_LINES'))
    JSON_DATASET_PATH = os.getenv('JSON_DATASET_PATH')
    FRAME_DATASET_PATH = os.getenv("FRAME_DATASET_PATH")

    JSON_GLOB_PATH = JSON_DATASET_PATH + '/*/*.json'

    converter = VILJsonConverter(
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        num_type_of_lines=NUM_TYPE_OF_LINES,
        json_glob_path=JSON_GLOB_PATH,
        final_shape=(FINAL_WIDTH, FINAL_HEIGHT),
        frame_dataset_path=FRAME_DATASET_PATH,
    )
    converter.exec()
    log.info('Done...')
    print('Done...')
