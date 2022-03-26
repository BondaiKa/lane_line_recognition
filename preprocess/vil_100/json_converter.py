import h5py
from typing import List, Dict, Tuple
import json
from os.path import join, dirname
import numpy as np
import glob
import os
from pathlib import Path
import cv2
from typing import Union
import logging

from lane_line_recognition.preprocess.vil_100.utils import Vil100Json, LANE_ID_FULL_LIST, LineType, get_valid_attribute
from lane_line_recognition.base import LaneLineRecognitionHDF5, AbstractConverter
from lane_line_recognition.preprocess.vil_100.fix_json_files import JsonReviewer
from lane_line_recognition.utils import one_hot_list_encoder

log = logging.getLogger(__name__)

Lane_type = List[Dict[str, int]]

polyline_width_type = np.ndarray
polyline_height_type = np.ndarray
labels_type = np.ndarray


class VIL100JsonConverter(AbstractConverter):
    def __init__(self,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 num_type_of_lines: int,
                 json_glob_path: str,
                 final_shape: Tuple[int, int],
                 frame_dataset_path: str,
                 slice_coefficient: int,
                 ):

        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        self.json_files = sorted(glob.glob(json_glob_path))
        self.files_count = len(self.json_files)
        self.frame_dataset_path = frame_dataset_path
        self.final_shape = final_shape
        self.slice_coefficient = slice_coefficient

        log.debug(f'VIL100JsonConverter params:\n{locals()}')

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
        points = np.pad(points, pad_width=(0, self.max_num_points * self.slice_coefficient * 2 - points.shape[0]),
                        mode='constant', constant_values=(-1,))
        points = tuple(np.split(points.reshape(-1, 2), 2, axis=1))
        polyline_widths, polyline_heights = points[0].flatten(), points[1].flatten()

        polyline_widths = polyline_widths[0::self.slice_coefficient]
        polyline_heights = polyline_heights[0::self.slice_coefficient]

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

        polyline_widths_output = np.empty(shape=(0, self.max_num_points)),
        polyline_heights_output = np.empty(shape=(0, self.max_num_points))
        labels_output = np.empty(shape=(0, self.num_type_of_lines))

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
            polyline_widths_output = np.append(polyline_widths_output, polyline_widths)
            polyline_heights_output = np.append(polyline_heights_output, polyline_heights)
            labels_output = np.append(labels_output, label)

        return (polyline_widths_output, polyline_heights_output), np.split(labels_output, self.max_lines_per_frame)

    def exec(self) -> None:
        """Convert and save json files to new hdf5 files"""
        for json_file_path in self.json_files:
            polylines, labels = self.get_data_from_file(json_file_path)

            label_1, label_2 = labels[0], labels[1]
            polyline_widths, polyline_heights = polylines[0], polylines[1]
            full_path_list = json_file_path.split('/')
            full_path_list[-3] = LaneLineRecognitionHDF5.root_folder
            root_path = full_path_list[:-1]
            frame_name = full_path_list[-1]

            Path(f"{'/'.join(root_path)}").mkdir(parents=True, exist_ok=True)

            with h5py.File(f"{'/'.join(root_path)}/{frame_name}.hdf5", "w") as f:
                grp = f.create_group(LaneLineRecognitionHDF5.group_name)
                grp.create_dataset(LaneLineRecognitionHDF5.polyline_widths_dataset_name, data=polyline_widths,
                                   dtype='float32')
                grp.create_dataset(LaneLineRecognitionHDF5.polyline_heights_dataset_name, data=polyline_heights,
                                   dtype='float32')
                grp.create_dataset(LaneLineRecognitionHDF5.label_1_dataset_name, data=label_1, dtype='int')
                grp.create_dataset(LaneLineRecognitionHDF5.label_2_dataset_name, data=label_2, dtype='int')


if __name__ == '__main__':
    from dotenv import load_dotenv

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
    SLICE_COEFFICIENT = int(os.getenv('SLICE_COEFFICIENT'))

    JSON_GLOB_PATH = JSON_DATASET_PATH + '/*/*.json'

    converter = VIL100JsonConverter(
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        num_type_of_lines=NUM_TYPE_OF_LINES,
        json_glob_path=JSON_GLOB_PATH,
        final_shape=(FINAL_WIDTH, FINAL_HEIGHT),
        frame_dataset_path=FRAME_DATASET_PATH,
        slice_coefficient=SLICE_COEFFICIENT,
    )
    converter.exec()
    log.info('Done...')
