from abc import ABC
import cv2
import logging
from typing import Union
import numpy as np

log = logging.getLogger(__name__)


class AbstractVideoHandler(ABC):
    """Base Video handler"""

    def __init__(self, camera_path: Union[str, int]):
        self.video = cv2.VideoCapture(camera_path)

    def _process(self):
        log.info("Start video processing")
        self.process()
        log.info("End video processing")

    def process(self):
        """start to capture video"""
        raise NotImplemented

    def release(self):
        """release camera"""
        self.video.release()


class RosVideoPureLaneLineRecognition(AbstractVideoHandler):
    """Process original frames witout preprocessing"""

    def process(self):
        """Process pure image processing

        Process frames without filters and perspective transformation
        """
        while self.video.isOpened():
            ret, frame_dis = self.video.read()
            ###
            # TODO @Karim: add process image by NN
            ###
            cv2.imshow('cap', frame_dis)
            cv2.destroyAllWindows()
        return


class RosVideoTransformedLaneLineRecognition(AbstractVideoHandler):
    """Apply filter and perspective transformation"""

    def process(self):
        """Process image with filter and transform image

        Process frames with operators and perspective transformation
        """
        while self.video.isOpened():
            ret, frame_dis = self.video.read()
            ###
            # TODO @Karim: add perspective transformation
            ###

            ###
            # TODO @Karim: apply filter
            ###

            ###
            # TODO @Karim: add process image by NN
            ###
            cv2.imshow('cap', frame_dis)
            cv2.destroyAllWindows()
        return


class FakeVideoPureLaneLineRecognition(AbstractVideoHandler):
    """Process fake frames witout preprocessing"""

    def process(self, ret: bool, frame_dis: np.ndarray):
        ###
        # TODO @Karim: add process image by NN
        ###

        return


class FakeVideoTransformedLaneLineRecognition(AbstractVideoHandler):
    """Apply filter and perspective transformation for fake frames"""

    def process(self, ret: bool, frame_dis: np.ndarray):
        ###
        # TODO @Karim: add perspective transformation
        ###

        ###
        # TODO @Karim: apply filter
        ###

        ###
        # TODO @Karim: add process image by NN
        ###

        return
