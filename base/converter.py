from abc import ABCMeta, abstractmethod


class AbstractConverter(metaclass=ABCMeta):
    @abstractmethod
    def get_data_from_file(self, json_path: str):
        raise NotImplementedError

    @abstractmethod
    def exec(self):
        raise NotImplementedError
