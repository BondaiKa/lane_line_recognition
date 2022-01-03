from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from typing import Tuple, List, Dict, Optional
import glob
import json
import random

from tensorflow.keras.utils import Sequence
from utils import one_hot_list_encoder
import logging

BASE_DIR = "VIL100/"
IMAGE_PATH = BASE_DIR + "JPEGImages/"
JSON_PATH = BASE_DIR + "Json/"

log = logging.getLogger(__name__)


class SimpleFrameGenerator(Sequence):
    """Sequence of frames generator

    Usage for training NN that could process independent
    frames without context window etc
    """

    def __init__(self,
                 num_type_of_lines=2,
                 max_num_points=91,
                 max_lines_per_frame=6,
                 rescale=1 / 255.,  # todo canny operator etc
                 batch_size: int = 64,
                 target_shape: Tuple[int, int] = (1280, 960),
                 shuffle: bool = False,
                 split: Optional[float] = None,
                 nb_channel: int = 3,  # todo: read about this param
                 frame_glob_path: str = "",
                 json_glob_path: str = ""):
        """
        :param max_lines_per_frame: maxinum number of lines per frame
        :param max_num_points: maximum number of points un one polyline
        :param num_type_of_lines: number of possible lines on road
        :param rescale:
        :param batch_size: batch size of the dataset
        :param target_shape: final size for NN input
        :param shuffle: shuffle flag of frames sequences
        :param split: split dataset to train/test
        :param nb_channel: grayscaled or RGB frames
        :param frame_glob_path: glob pattern of frames
        :param json_glob_path: glob pattern path of jsons
        """

        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        self.rescale = rescale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel

        self.files = sorted(glob.glob(frame_glob_path))
        self.files_count = len(self.files)

        self.json_files = sorted(glob.glob(json_glob_path))
        self.num_json_files = len(self.json_files)

        log.info(f"Number of files: {self.files_count}.")

        if self.files_count != self.num_json_files:
            log.error(f"Datasaet files error"
                      f"Number of frames: ({self.files_count}). Number of jsons({self.num_json_files})")
            raise FileNotFoundError(
                f"Numbers of frames and jsons are not equal!")

        if split and 0.0 < split < 1.0:
            ######
            # TODO split data to train and test dataset
            ######
            self.files_count = int(split * self.files_count)
            self.files = self.files[:self.files_count]
            self.json_files = self.json_files[:self.files_count]

        if shuffle:
            temp = list(zip(self.files, self.json_files))
            random.shuffle(temp)
            self.files, self.json_files = zip(*temp)
            del temp

    def __len__(self):
        return math.ceil(self.files_count / self.batch_size)

    def __get_polyline_with_label(self, lane: dict) -> np.ndarray:
        """Get array from points list"""
        res = np.array(
            lane["points"]).flatten()
        res = np.pad(res, pad_width=(0, self.max_num_points * 2 - res.shape[0]))
        res = np.hstack((res, one_hot_list_encoder(lane.get('label', 0), self.num_type_of_lines)))
        return res

    def __get_polyline_from_file(self, json_path) -> np.ndarray:
        """
        Get all Polygonal chains from json file
        :param json_path: path of json file
        :return: right points for frame
        """
        with open(json_path) as f:
            polylines: List[Dict[str, int]] = json.load(f)["annotations"]["lane"]
            # TODO @Karim: check another params in json files like "occlusion"
            res = np.vstack(list(map(lambda lane: self.__get_polyline_with_label(lane=lane), polylines)))
            empty_lines = np.zeros(
                shape=(self.max_lines_per_frame - res.shape[0], self.max_num_points * 2 + self.num_type_of_lines))
            return np.vstack((res, empty_lines))

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        # previous_frames = None
        # diff = self.nbframe - idx - 1
        # if diff > 0:
        #################################################
        # TODO @Karim: remove cnn-lstm preparation code
        #################################################
        # TODO: @Karim include batch
        # TODO @Karim: check that diff video frames will not be included in one input (batch or input ????)
        # previous_frames = np.zeros(shape=(diff * self.target_shape[0] * self.target_shape[1]))
        # else:

        batch_frames_path = self.files[idx * self.batch_size:
                                       (idx + 1) * self.batch_size]
        batch_json_path = self.json_files[idx * self.batch_size:
                                          (idx + 1) * self.batch_size]
        polylines = np.array(list(map(lambda x: self.__get_polyline_from_file(x),
                                      batch_json_path)))
        # TODO @Karim: test that polylines relate to right frames
        return np.array([
            resize(imread(file_name) * self.rescale, self.target_shape) for file_name in batch_frames_path]
        ), polylines


if __name__ == "__main__":
    image_glob_path = IMAGE_PATH + '/*/*.jpg'
    json_glob_path = JSON_PATH + '/*/*.json'
    frame_generator = SimpleFrameGenerator(frame_glob_path=image_glob_path, json_glob_path=json_glob_path,
                                           split=0.8, shuffle=False)

    for item in frame_generator:
        print([x.shape for x in item[1]])
