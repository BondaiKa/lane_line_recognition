from abc import ABCMeta, abstractmethod


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
