from abc import ABC
import cv2
import logging
from typing import Union

log = logging.getLogger(__name__)


class AbstractVideoHandler(ABC):

    def __init__(self, camera_path: Union[str, int]):
        self.video = cv2.VideoCapture(camera_path)

    def process(self):
        raise NotImplemented

    def release(self):
        self.video.release()


class RosVideoHandler(AbstractVideoHandler):

    def process(self):
        log.info("Start video processing")
        ret, frame_dis = self.video.read()

        ###
        # TODO @Karim: add old_code process image by nn
        ###

        log.info("End video processing")


class FakeCameraHandler(AbstractVideoHandler):

    def process(self):
        log.info("Start video processing")

        while self.video.isOpened():
            ret, frame_dis = self.video.read()
            log.info(f"Ret value: {ret}")
            cv2.imshow('cap', frame_dis)
            cv2.waitKey(0)

            ###
            # TODO @Karim: add nn line recognition
            ###

        log.info("End video processing")
        cv2.destroyAllWindows()
