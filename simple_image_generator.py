from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from typing import Tuple, List, Dict, Iterable
import glob
import json

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
                 rescale=1 / 255.,  # todo canny operator etc
                 nbframe: int = 4,
                 batch_size: int = 64,
                 target_shape: Tuple[int, int] = (1280, 960),
                 shuffle: bool = True,
                 transformation: ImageDataGenerator = None,
                 split: float = None,  # todo: wtf?
                 nb_channel: int = 3,  # todo: read about this param
                 frame_glob_path: str = "",
                 json_glob_path: str = ""):
        """
        :param num_type_of_lines: number of possible lines on road
        :param rescale:
        :param nbframe: number of frame to return for each sequence
        :param batch_size: batch size of the dataset
        :param target_shape: final size for NN input
        :param shuffle: shuffle flag of frames sequences
        :param transformation: ImageDataGenerator
        :param split: split dataset to train/test
        :param nb_channel: grayscaled or RGB frames
        :param frame_glob_path: glob pattern of frames
        :param json_glob_path: glob pattern path of jsons
        """

        self.num_type_of_lines = num_type_of_lines
        self.rescale = rescale
        self.batch_size = batch_size
        self.nbframe = nbframe
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation

        ######
        # TODO split data to train and test dataset
        ######

        ######
        # TODO: shuffle dataset
        ######

        # build indexes
        self.files = glob.glob(frame_glob_path)
        self.files_count = len(self.files)

        self.json_files = glob.glob(json_glob_path)
        self.num_json_files = len(self.json_files)

        log.info(f"Number of files: {self.files_count}.")

        if self.files_count != self.num_json_files:
            log.error(f"Datasaet files error"
                      f"Number of frames: ({self.files_count}). Number of jsons({self.num_json_files})")
            raise FileNotFoundError(
                f"Numbers of frames and jsons are not equal!")

    def __len__(self):
        return math.ceil(self.files_count / self.batch_size)

    def __get_polyline_from_file(self, json_path) -> np.ndarray:
        """
        Get all Polygonal chains from json file
        :param json_path: path of json file
        :return: right points for frame
        """
        with open(json_path) as f:
            polylines: List[Dict[str, int]] = json.load(f)["annotations"]["lane"]
            # TODO @Karim: check another params in json files like "occlusion"
            return np.array(list(map(lambda x: np.array(
                x["points"] + (one_hot_list_encoder(x.get('label', 0), self.num_type_of_lines))).flatten(),
                                     polylines)))

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
    frame_generator = SimpleFrameGenerator(frame_glob_path=image_glob_path, json_glob_path=json_glob_path)
