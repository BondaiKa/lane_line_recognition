import logging

from fake_video_handler import FakeVideoHandler

log = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s", level=logging.DEBUG)

CAMERA_PATH = "../dataset/IMG_0261.MOV"

if __name__ == "__main__":
    log.info("Start working...")

    camera = FakeVideoHandler(camera_path=CAMERA_PATH)
    camera.process()
    camera.release()

    log.info("End working...")
