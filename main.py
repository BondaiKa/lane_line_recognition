import logging
import os

from fake_video_handler import FakeVideoHandler
from lane_line_recognition.base import FrameHandler

log = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s", level=logging.DEBUG)

log.info('load environment variables...')
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    log.info("Start working...")

    CAMERA_PATH = os.getenv('CAMERA_PATH')
    NEURAL_NETWORK_WEIGHTS_MODEL_PATH = os.getenv('NEURAL_NETWORK_WEIGHTS_MODEL_PATH')
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT'))
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(os.getenv('NUM_TYPE_OF_LINES'))

    camera = FakeVideoHandler(camera_path=CAMERA_PATH)
    frame_handler = FrameHandler(
        model_weights_path=NEURAL_NETWORK_WEIGHTS_MODEL_PATH,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        num_type_of_lines=NUM_TYPE_OF_LINES
    )

    camera.process(frame_handler=frame_handler)
    camera.release()

    log.info("End working...")
