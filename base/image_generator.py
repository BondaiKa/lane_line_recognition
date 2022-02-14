from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Callable, Type
import glob
import logging

log = logging.getLogger(__name__)


class AbstractFrameGenerator(metaclass=ABCMeta):
    @abstractmethod
    def get_data_from_file(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_frame_from_file(self, frame_path: str):
        raise NotImplementedError


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
                 frame_glob_path: str,
                 json_hdf5_glob_path: str):
        """
        :param validation_split: split for train/validation sets
        :param frame_glob_path: glob pattern of frames
        :param json_hdf5_glob_path: glob pattern path of jsons
        """
        self.validation_split = validation_split
        self.__frame_glob_path = frame_glob_path
        self.__json_hdf5_glob_path = json_hdf5_glob_path

    def preprocess_files(self, subset: str, number_files: int) -> Tuple[List[str], List[str]]:
        """
        Sort and validate files for further loading and sending to NN

        :param subset: 'training' or 'validation'
        :param number_files: restrict max number of files from dataset
        :return: list of sorted and sliced files and json files
        """

        files = sorted(glob.glob(self.__frame_glob_path))
        log.info(f"Number of files in dataset: {len(files)}. Using in training/validation: {number_files}")
        files = files[:number_files]

        json_files = sorted(glob.glob(self.__json_hdf5_glob_path))[:number_files]
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
                            number_files: int,
                            *args, **kwargs) -> Callable:
        files, json_files = self.preprocess_files(subset=subset,
                                                  number_files=number_files)
        return self.get_generator()(files=files,
                                    json_files=json_files,
                                    *args, **kwargs)
