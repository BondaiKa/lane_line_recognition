from abc import ABCMeta, abstractmethod
import glob
from typing import Tuple


class AbstractConverter(metaclass=ABCMeta):

    @abstractmethod
    def get_data_from_file(self, json_path: str):
        raise NotImplementedError

    @abstractmethod
    def exec(self):
        raise NotImplementedError


class AbstractFrameGenerator(metaclass=ABCMeta):
    @abstractmethod
    def get_data_from_file(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_frame_from_file(self, frame_path: str):
        raise NotImplementedError


class AbstractFrameGeneratorCreator(metaclass=ABCMeta):
    @abstractmethod
    def flow_from_directory(self,
                            subset: str,
                            shuffle: bool,
                            number_files: int,
                            *args, **kwargs):
        raise NotImplementedError
