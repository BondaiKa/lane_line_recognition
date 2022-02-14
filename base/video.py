from abc import ABC
import cv2
from typing import Union

import logging

log = logging.getLogger(__name__)


class AbstractVideoHandler(ABC):
    """Base Video handler"""

    def __init__(self, camera_path: Union[str, int]):
        self.video = cv2.VideoCapture(camera_path)

    def process(self):
        """start to capture video"""
        raise NotImplemented

    def release(self):
        """release camera"""
        self.video.release()

    def exec(self):
        log.info("Start video processing...")
        self.process()
        log.info("End video processing...")
