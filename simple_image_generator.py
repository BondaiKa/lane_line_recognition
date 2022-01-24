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
from vil_100_utils import get_valid_attribute, LANE_ID_FULL_LIST
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
                 num_type_of_lines=11,
                 max_num_points=91,
                 max_lines_per_frame=8,
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

    def __get_polyline_with_label(self, lane: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get array from points list"""
        points = np.array(
            lane["points"]).flatten()
        points = np.pad(points, pad_width=(0, self.max_num_points * 2 - points.shape[0]))
        # TODO @Karim: remember below `label.get(label)` is index 1,2,3,4
        label = get_valid_attribute(lane.get('attribute', 1))
        labels = one_hot_list_encoder(label, self.num_type_of_lines)
        return points, labels

    def __get_polyline_and_label_from_file(self, json_path: str) -> np.ndarray:
        """
        Get all Polygonal chains from json file and label of line
        :param json_path: path of json file
        :return: right points for frame
        """
        with open(json_path) as f:
            lanes: List[Dict[str, int]] = json.load(f)["annotations"]["lane"]
            lanes = sorted(lanes, key=lambda lane: lane['lane_id'])

            if lanes:
                polylines, labels = list(), list()
                # TODO @Karim: check another params in json files like "occlusion"
                exist_lane = [x['lane_id'] for x in lanes]
                missed_lane = LANE_ID_FULL_LIST - set(exist_lane)

                for lane_id in range(1, self.max_lines_per_frame + 1):
                    if lane_id in missed_lane:
                        points = np.zeros(shape=(self.max_num_points * 2))
                        label = one_hot_list_encoder(0, self.num_type_of_lines)
                    else:
                        points, label = self.__get_polyline_with_label(lane=lanes[exist_lane.index(lane_id)])

                    if lane_id % 2 == 0:
                        polylines.append(points)
                        labels.append(label)
                    else:
                        polylines = [points] + polylines
                        labels = [label] + labels

                return np.concatenate(polylines), labels
            else:
                empty_label = one_hot_list_encoder(0, self.num_type_of_lines)
                polylines_empty_shape = self.max_lines_per_frame * self.max_num_points * 2
                return np.zeros(shape=(polylines_empty_shape)), map(lambda x: x, empty_label)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        batch_frames_path = self.files[idx * self.batch_size:
                                       (idx + 1) * self.batch_size]
        batch_json_path = self.json_files[idx * self.batch_size:
                                          (idx + 1) * self.batch_size]
        polyline_list, list_of_list_label = list(), list([[],[],[],[],[],[],[],[]])

        for json in batch_json_path:
            polylines, *list_label = self.__get_polyline_and_label_from_file(json)
            polyline_list.append(polylines)

            for idx, label in enumerate(list_label[0]):
                list_of_list_label[idx].append(label)

        polylines_output = np.vstack(polyline_list)
        labels_output = tuple(map(lambda x: np.vstack(x), list_of_list_label))

        return np.array([
            resize(imread(file_name) * self.rescale, self.target_shape) for file_name in
            batch_frames_path]), (polylines_output,) + labels_output


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
                            shuffle: bool = True, number_files: int = 2000, *args, **kwargs) -> SimpleFrameGenerator:
        """
        Get generator for subset
        :param subset: 'training' or 'validation'
        :param shuffle: flag for shuffling
        :param number_files: rectrict max number of files from dataset
        :param args: args for specific dataset
        :param kwargs: kwargs for specific dataset
        :return: Specific generator for specific subset
        """

        files = sorted(glob.glob(self.__frame_glob_path))
        log.info(f"Number of files in dataset: {len(files)}. Using in training/validation: {number_files}")
        files = files[:number_files]

        json_files = sorted(glob.glob(self.__json_glob_path))[:number_files]
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
    BATCH_SIZE = 32
    image_glob_path = IMAGE_PATH + '/*/*.jpg'
    json_glob_path = JSON_PATH + '/*/*.json'
    data_gen = SimpleFrameDataGen(validation_split=0.2, frame_glob_path=image_glob_path, json_glob_path=json_glob_path)
    train_generator = data_gen.flow_from_directory(subset='training', shuffle=True,batch_size = BATCH_SIZE)
    validation_generator = data_gen.flow_from_directory(subset='validation', shuffle=True,batch_size = BATCH_SIZE)

    for item in train_generator:
        print(item[1][0].shape)
        print(item[1][1].shape)
        print(item[1][2].shape)
        print(item[1][3].shape)
        print(item[1][4].shape)
        print(item[1][5].shape)
        print(item[1][6].shape)
        print(item[1][7].shape)
        print(item[1][8].shape)
        break

    # for item in validation_generator:
    #     print([x.shape for x in item[1]])
    #     break
