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
                 rescale=1 / 255.,  # TODO @Karim: include and use later
                 batch_size: int = 8,
                 target_shape: Tuple[int, int] = (1280, 960),
                 shuffle: bool = False,
                 nb_channel: int = 3,  # TODO: Use rgb later
                 files: Optional[List[str]] = None,
                 json_files: Optional[List[str]] = None):
        """
        :param subset: training or validation data
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
        self.files = files
        self.json_files = json_files
        self.files_count = len(self.files)

        if shuffle:
            temp = list(zip(self.files, self.json_files))
            random.shuffle(temp)
            self.files, self.json_files = zip(*temp)

    def __len__(self):
        return math.ceil(self.files_count / self.batch_size)

    def __get_polyline_with_label(self, lane: dict) -> np.ndarray:
        """Get array from points list"""
        res = np.array(
            lane["points"]).flatten()
        res = np.pad(res, pad_width=(0, self.max_num_points * 2 - res.shape[0]))
        res = np.hstack((res, one_hot_list_encoder(lane.get('label', 0), self.num_type_of_lines)))
        return res

    def __get_polyline_from_file(self, json_path: str) -> np.ndarray:
        """
        Get all Polygonal chains from json file
        :param json_path: path of json file
        :return: right points for frame
        """
        with open(json_path) as f:
            polylines: List[Dict[str, int]] = json.load(f)["annotations"]["lane"]
            if polylines:
                # TODO @Karim: check another params in json files like "occlusion"
                res = np.vstack(list(map(lambda lane: self.__get_polyline_with_label(lane=lane), polylines)))
                empty_lines = np.zeros(
                    shape=(self.max_lines_per_frame - res.shape[0], self.max_num_points * 2 + self.num_type_of_lines),
                    dtype=np.float32)
                return np.vstack(
                    (res, empty_lines)).flatten()  # todo: rewrite to return flatten 1, 284 * 6 without flatten
            else:
                return np.zeros(shape=(self.max_lines_per_frame * (self.max_num_points * 2 + self.num_type_of_lines)))

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        batch_frames_path = self.files[idx * self.batch_size:
                                       (idx + 1) * self.batch_size]
        batch_json_path = self.json_files[idx * self.batch_size:
                                          (idx + 1) * self.batch_size]
        polylines = np.array(list(map(lambda x: self.__get_polyline_from_file(x),
                                      batch_json_path)))
        return np.array([
            resize(imread(file_name) * self.rescale, self.target_shape) for file_name in batch_frames_path]), polylines


class SimpleFrameDataGen:
    TRAINING = 'training'
    VALIDATION = 'validation'

    __reverse_dataset_type = {
        TRAINING: VALIDATION,
        VALIDATION: TRAINING
    }
    __dataset = {}

    def __init__(self,
                 rescale=1 / 255.,
                 validation_split: Optional[float] = None,
                 frame_glob_path: str = "",
                 json_glob_path: str = ""):
        """
        :param validation_split: split for train/validation sets
        :param rescale:
        :param frame_glob_path: glob pattern of frames
        :param json_glob_path: glob pattern path of jsons
        """
        self.rescale = rescale
        self.validation_split = validation_split

        self.__frame_glob_path = frame_glob_path
        self.__json_glob_path = json_glob_path

    def flow_from_directory(self, subset: str = TRAINING,
                            shuffle: bool = True, *args, **kwargs) -> SimpleFrameGenerator:
        """
        Get generator for subset
        :param subset: 'training' or 'validation'
        :param shuffle: flag for shuffling
        :param args: args for specific dataset
        :param kwargs: kwargs for specific dataset
        :return: Specific generator for specific subset
        """

        files = sorted(glob.glob(self.__frame_glob_path))
        json_files = sorted(glob.glob(self.__json_glob_path))
        files_count = len(files)
        json_files_count = len(json_files)

        if files_count != json_files_count:
            log.error(f"Dataset files error"
                      f"Number of frames: ({files_count}). "
                      f"Number of jsons({json_files_count}")
            raise FileNotFoundError(
                f"Numbers of frames and jsons are not equal!")

        if not self.__reverse_dataset_type.get(subset):
            log.error(f'Wrong subset value: "{subset}"')
            raise ValueError(f'Wrong type of subset - {subset}. '
                             f'Available types: {self.__reverse_dataset_type.keys()}')

        if self.validation_split and 0.0 < self.validation_split < 1.0:
            split = int(files_count * (1 - self.validation_split))
            if subset == self.TRAINING:
                files = files[:split]
                json_files = json_files[:split]
            else:
                files = files[split:]
                json_files = json_files[split:]

        return SimpleFrameGenerator(rescale=self.rescale,
                                    files=files,
                                    shuffle=shuffle,
                                    json_files=json_files,
                                    *args, **kwargs)


if __name__ == "__main__":
    image_glob_path = IMAGE_PATH + '/*/*.jpg'
    json_glob_path = JSON_PATH + '/*/*.json'
    data_gen = SimpleFrameDataGen(validation_split=0.2, frame_glob_path=image_glob_path, json_glob_path=json_glob_path)
    train_generator = data_gen.flow_from_directory(subset='training', shuffle=True)
    validation_generator = data_gen.flow_from_directory(subset='validation', shuffle=True)

    for item in train_generator:
        print([x.shape for x in item[1]])
        break

    for item in validation_generator:
        print([x.shape for x in item[1]])
        break
