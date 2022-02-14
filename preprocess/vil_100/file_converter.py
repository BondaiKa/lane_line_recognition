import h5py
from typing import List, Dict, Tuple
import json
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

from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)

Lane_type = List[Dict[str, int]]


class VILLJsonConverter:

    def __init__(self,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 num_type_of_lines: int,
                 json_glob_path: str,
                 final_shape: Tuple[int, int],
                 frame_dataset_path: str,
                 rescale_polyline_coef: float,
                 ):

        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        self.json_files = sorted(glob.glob(json_glob_path))
        self.files_count = len(self.json_files)
        self.frame_dataset_path = frame_dataset_path
        self.final_shape = final_shape
        self.rescale_polyline_coef = rescale_polyline_coef

    def __rescale_coordinates(self, row: np.ndarray, initial_width: int, initial_height: int) -> np.ndarray:
        return row[0] / initial_width * self.final_shape[0] * self.rescale_polyline_coef, \
               row[1] / initial_height * self.final_shape[1] * self.rescale_polyline_coef

    def __rescale_polylines(self, polylines: np.ndarray, initial_width: int, initial_height: int) -> np.ndarray:
        """Rescale coordinates due to new frame resolution"""
        return np.apply_along_axis(self.__rescale_coordinates, axis=1, arr=polylines,
                                   initial_width=initial_width,
                                   initial_height=initial_height
                                   )

    def __get_polyline_with_label(self, lane: dict, initial_width: int, initial_height: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """Get array from points list"""
        points = np.array(
            lane[Vil100Json.POINTS])
        points = self.__rescale_polylines(points, initial_width=initial_width, initial_height=initial_height).flatten()
        points = np.pad(points, pad_width=(0, self.max_num_points * 2 - points.shape[0]),
                        mode='constant', constant_values=(-1,))
        # TODO @Karim: remember below `label.get(label)` is index 1,2,3,4
        label = get_valid_attribute(lane.get(Vil100Json.ATTRIBUTE, LineType.NO_LINE))
        labels = one_hot_list_encoder(label, self.num_type_of_lines)
        return points, labels

    def __get_polyline_and_label_from_file(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve from json file polylines and labels and format to nn input

        :param json_path: json file path
        :return: frame and tuple of labels
        """
        with open(json_path) as f:
            json_file: Dict[str, Union[int, dict]] = json.load(f)
            image_path = json_file[Vil100Json.INFO][Vil100Json.IMAGE_PATH]
            full_path = self.frame_dataset_path + '/' + image_path
            frame = cv2.imread(full_path)
            height, width = frame.shape[0], frame.shape[1]
            json_file = JsonReviewer.fix_json_file(
                json_file=json_file,
                frame_real_height=height,
                frame_real_width=width,
                frame_path=image_path
            )

            width = json_file[Vil100Json.INFO][Vil100Json.WIDTH]
            height = json_file[Vil100Json.INFO][Vil100Json.HEIGHT]

            lanes: Lane_type = json_file[Vil100Json.ANNOTATIONS][Vil100Json.LANE]
            lanes = sorted(lanes, key=lambda lane: lane[Vil100Json.LANE_ID])

            if lanes:
                polylines, labels = np.array([]), np.array([])
                # TODO @Karim: check another params in json files like "occlusion"
                exist_lane = [x[Vil100Json.LANE_ID] for x in lanes]
                missed_lane = LANE_ID_FULL_LIST - set(exist_lane)

                for lane_id in range(1, self.max_lines_per_frame + 1):
                    if lane_id in missed_lane:
                        points = np.zeros(shape=(self.max_num_points * 2))
                        label = one_hot_list_encoder(LineType.NO_LINE, self.num_type_of_lines)
                    else:
                        points, label = self.__get_polyline_with_label(
                            lane=lanes[exist_lane.index(lane_id)],
                            initial_width=width,
                            initial_height=height
                        )

                    if lane_id % 2 == 0:
                        polylines = np.append(polylines, points)
                        labels = np.append(labels, label)
                    else:
                        polylines = np.insert(polylines, 0, points)
                        labels = np.insert(labels, 0, label)

                return polylines, labels
            else:
                empty_label = one_hot_list_encoder(LineType.NO_LINE, self.num_type_of_lines)
                polylines_empty_shape = self.max_lines_per_frame * self.max_num_points * 2
                return np.zeros(shape=polylines_empty_shape), np.array(
                    [empty_label for x in range(self.max_lines_per_frame)]).flatten()

    def exec(self) -> None:
        """Convert and save json files to new hdf5 files"""
        for json_file_path in self.json_files:
            polylines, labels = self.__get_polyline_and_label_from_file(json_file_path)

            full_path_list = json_file_path.split('/')
            full_path_list[-3] = VIL100HDF5.ROOT_FOLDER
            root_path = full_path_list[:-1]
            frame_name = full_path_list[-1]

            Path(f"{'/'.join(root_path)}").mkdir(parents=True, exist_ok=True)

            with h5py.File(f"{'/'.join(root_path)}/{frame_name}.hdf5", "w") as f:
                grp = f.create_group(VIL100HDF5.GROUP_NAME)
                grp.create_dataset(VIL100HDF5.POLYLINES_DATASET_NAME, data=polylines, dtype='float32')
                grp.create_dataset(VIL100HDF5.LABELS_DATASET_NAME, data=labels, dtype='float32')


if __name__ == '__main__':
    load_dotenv()
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT'))
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(os.getenv('NUM_TYPE_OF_LINES'))
    JSON_PATH = os.getenv('JSON_DATASET_PATH')
    FRAME_DATASET_PATH = os.getenv("FRAME_DATASET_PATH")
    RESCALE_POLYLINE_COEFFICIENT = float(os.getenv("RESCALE_POLYLINE_COEFFICIENT"))

    final_shape = (CAMERA_WIDTH, CAMERA_HEIGHT)
    json_glob_path = JSON_PATH + '/*/*.json'

    converter = VILLJsonConverter(
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        num_type_of_lines=NUM_TYPE_OF_LINES,
        json_glob_path=json_glob_path,
        final_shape=final_shape,
        frame_dataset_path=FRAME_DATASET_PATH,
        rescale_polyline_coef=RESCALE_POLYLINE_COEFFICIENT,
    )
    converter.exec()
    log.info('Done...')
