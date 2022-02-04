import h5py
from typing import List, Dict, Tuple, Optional
import json
from vil_100_utils import Vil100Json, LANE_ID_FULL_LIST, LineType, get_valid_attribute
import numpy as np
from utils import one_hot_list_encoder
import glob
import os
from pathlib import Path

from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)


class VILLJsonConverter:

    def __init__(self,
                 max_lines_per_frame: int,
                 max_num_points: int,
                 num_type_of_lines: int,
                 frame_glob_path: str = None,
                 json_glob_path: str = None, ):

        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        # self.files = sorted(glob.glob(frame_glob_path))
        self.json_files = sorted(glob.glob(json_glob_path))
        self.files_count = len(self.json_files)

    def __get_polyline_with_label(self, lane: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get array from points list"""
        points = np.array(
            lane[Vil100Json.POINTS]).flatten()
        points = np.pad(points, pad_width=(0, self.max_num_points * 2 - points.shape[0]))
        # TODO @Karim: remember below `label.get(label)` is index 1,2,3,4
        label = get_valid_attribute(lane.get(Vil100Json.ATTRIBUTE, LineType.NO_LINE))
        labels = one_hot_list_encoder(label, self.num_type_of_lines)
        return points, labels

    def __get_polyline_and_label_from_file(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(json_path) as f:
            lanes: List[Dict[str, int]] = json.load(f)[Vil100Json.ANNOTATIONS][Vil100Json.LANE]
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
                        points, label = self.__get_polyline_with_label(lane=lanes[exist_lane.index(lane_id)])

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
        for json_file_path in self.json_files:
            polylines, labels = self.__get_polyline_and_label_from_file(json_file_path)

            video_name, frame_name = json_file_path.split('/')[-2:]

            Path(f"hdf5/{video_name}").mkdir(parents=True, exist_ok=True)

            with h5py.File(f"hdf5/{video_name}/{frame_name}.hdf5", "w") as f:
                grp = f.create_group('frame_polylines_labels')
                grp.create_dataset("polylines", data=polylines, dtype='int32')
                grp.create_dataset("labels", data=labels, dtype='int32')


if __name__ == '__main__':
    load_dotenv()
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT'))
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(os.getenv('NUM_TYPE_OF_LINES'))
    JSON_PATH = os.getenv('JSON_DATASET_PATH')

    json_glob_path = JSON_PATH + '/*/*.json'

    converter = VILLJsonConverter(
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        num_type_of_lines=NUM_TYPE_OF_LINES,
        json_glob_path=json_glob_path,
    )
    converter.exec()
