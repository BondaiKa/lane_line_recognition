import logging
import os

from fake_video_handler import RosVideoHandler, FakeVideoMinimumLaneLineRecognition

log = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG)

CAMERA_PATH = "../dataset/IMG_0261.MOV"

if __name__ == "__main__":
    log.info("Start working...")

    ###
    # TODO @Karim: add rospy code
    ###

    camera = FakeVideoMinimumLaneLineRecognition(camera_path=CAMERA_PATH)
    camera.process()
    camera.release()

    log.info("End working...")
