from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional, Type, Callable
from functools import reduce
import glob
import logging
import tensorflow as tf
import numpy as np
import random
import math

log = logging.getLogger(__name__)


class AbstractFrameGenerator(metaclass=ABCMeta):

    def __init__(self,
                 num_type_of_lines,
                 max_lines_per_frame,
                 max_num_points,
                 batch_size: int,
                 final_shape: Tuple[int, int],
                 shuffle: bool,
                 files: Optional[List[str]],
                 json_files: Optional[List[str]],
                 rescale=1 / 255.,
                 color_mode: str = 'grayscale',
                 ):
        """
        :param max_lines_per_frame: maximum number of lines per frame
        :param max_num_points: maximum number of points un one polyline
        :param num_type_of_lines: number of possible lines on road
        :param rescale:
        :param batch_size: batch size of the dataset
        :param final_shape: final size for NN input
        :param shuffle: shuffle flag of frames sequences
        :param color_mode: `grayscale` or `rgb` color reading frame mod
        :param json_files: list of json files that contain info about a frame
        """
        self.max_lines_per_frame = max_lines_per_frame
        self.max_num_points = max_num_points
        self.num_type_of_lines = num_type_of_lines
        self.rescale = rescale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.final_shape = final_shape
        self.color_mode = color_mode
        self.files = files
        self.json_files = json_files
        self.files_count = len(self.files)

        if shuffle:
            temp = list(zip(self.files, self.json_files))
            random.shuffle(temp)
            self.files, self.json_files = zip(*temp)

    def __len__(self):
        return math.ceil(self.files_count / self.batch_size)

    @abstractmethod
    def get_data_from_file(self, *args, **kwargs):
        raise NotImplementedError

    def get_frame_from_file(self, frame_path: str) -> np.ndarray:
        frame = tf.keras.utils.load_img(frame_path,
                                        color_mode=self.color_mode,
                                        target_size=(self.final_shape[1], self.final_shape[0])
                                        )
        frame = tf.keras.preprocessing.image.img_to_array(frame)
        frame = frame * self.rescale
        frame = np.expand_dims(frame, 0)
        return frame


class AbstractFrameGeneratorCreator(metaclass=ABCMeta):
    TRAINING = 'training'
    VALIDATION = 'validation'

    __reverse_dataset_type = {
        TRAINING: VALIDATION,
        VALIDATION: TRAINING
    }
    __dataset = {}

    def __init__(self,
                 validation_split: float,
                 frame_glob_path: List[str],
                 json_hdf5_glob_path: List[str]):
        """
        :param validation_split: split for train/validation sets
        :param frame_glob_path: glob pattern of frames
        :param json_hdf5_glob_path: glob pattern path of jsons
        """
        self.validation_split = validation_split
        self.__frame_glob_path = frame_glob_path
        self.__json_hdf5_glob_path = json_hdf5_glob_path

    def preprocess_files(self, subset: str, number_files: Optional[int]) -> Tuple[List[str], List[str]]:
        """
        Sort and validate files for further loading and sending to NN

        :param subset: 'training' or 'validation'
        :param number_files: restrict max number of files from dataset
        :return: list of sorted and sliced files and json files
        """
        log.debug(f'Generator params: {locals()}')

        files = []
        for glob_path in self.__frame_glob_path:
          files.extend(glob.glob(glob_path))

        log.info(
            f"Number of files in dataset: {len(files)}."
            f"Using in training/validation: {str(number_files) if number_files else 'all files'}"
        )

        json_files = []
        for glob_path in self.__json_hdf5_glob_path:
          json_files.extend(glob.glob(glob_path))

        if number_files:
          files = files[:number_files]
          json_files = json_files[:number_files]

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

        return files, json_files

    @abstractmethod
    def get_generator(self) -> Type:
        raise NotImplementedError

    def flow_from_directory(self,
                            subset: str,
                            number_files: Optional[int]=None,
                            *args, **kwargs) -> Callable:
        files, json_files = self.preprocess_files(subset=subset,
                                                  number_files=number_files)
        return self.get_generator()(files=files,
                                    json_files=json_files,
                                    *args, **kwargs)
