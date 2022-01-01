from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from typing import Optional, List
import glob

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "VIL100/"
IMAGE_PATH = BASE_DIR + "JPEGImages/"
JSON_PATH = BASE_DIR + "Json/"

import logging

log = logging.getLogger(__name__)


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

# Every Sequence must implement the __getitem__ and the __len__ methods.
# If you want to modify your dataset between epochs you may implement
# on_epoch_end. The method __getitem__ should return a complete batch.

class VideoFrameGenerator(Sequence):
    """Sequence of frames generator"""

    def __init__(self,
                 rescale=1 / 255.,  # todo canny operator etc
                 nbframe: int = 4,
                 batch_size: int = 64,
                 target_shape: tuple = (1280, 960),
                 shuffle: bool = True,
                 transformation: ImageDataGenerator = None,
                 split: float = None,  # todo: wtf?
                 nb_channel: int = 3,  # todo: read about this param
                 frame_glob_path: str = "",
                 json_glob_path: str = ""):
        """
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
        self.indexes = np.arange(self.files_count)
        self.polylines = glob.glob(json_glob_path)

        log.info(f"Number of files: {self.files_count}.")

        if self.files_count != len(self.polylines):
            log.error(f"Datasaet files error"
                      f"Number of frames: ({self.files_count}). Number of jsons({len(self.polylines)})")
            raise FileNotFoundError(
                f"Numbers of frames and jsons are not equal!")

    def __len__(self):
        return math.ceil(self.files_count / self.batch_size)

    def __getitem__(self, idx):
        previous_frames = None
        diff = self.nbframe - idx - 1
        if diff > 0:

            #TODO: @Karim include batch
            #TODO @Karim: check that diff video frames will not be included in one input (batch or input ????)
            previous_frames = np.zeros(shape=(diff * self.target_shape[0] * self.target_shape[1]))
        else:

        return  # np.array([num of images]), np.array(todo equal one json polyline)

    def on_epoch_end(self):
        """Method called at the end of every epoch."""
        pass


if __name__ == "__main__":
    image_glob_path = IMAGE_PATH + '/*/*.jpg'
    json_glob_path = JSON_PATH + '/*/*.json'
    frame_generator = VideoFrameGenerator(frame_glob_path=image_glob_path, json_glob_path=json_glob_path)
