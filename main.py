import logging
import os

from lane_line_recognition.fake_video_handler import FakeVideoHandler
from lane_line_recognition.base import FrameHandler

log = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s", level=logging.DEBUG)

log.info('load environment variables...')
from dotenv import load_dotenv

if __name__ == "__main__":
    log.info("Start working...")

    load_dotenv()
    CAMERA_PATH = os.getenv('CAMERA_PATH')
    POLYLINE_NEURAL_NETWORK_MODEL_PATH = os.getenv('POLYLINE_NEURAL_NETWORK_MODEL_PATH')
    LABEL_NEURAL_NETWORK_MODEL_PATH = os.getenv('LABEL_NEURAL_NETWORK_MODEL_PATH')

    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT'))
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(os.getenv('NUM_TYPE_OF_LINES'))
    NEURAL_NETWORK_WIDTH = int(os.getenv('NEURAL_NETWORK_WIDTH'))
    NEURAL_NETWORK_HEIGHT = int(os.getenv('NEURAL_NETWORK_HEIGHT'))


    camera = FakeVideoHandler(camera_path=CAMERA_PATH)
    frame_handler = FrameHandler(
        polyline_model_path=POLYLINE_NEURAL_NETWORK_MODEL_PATH,
        label_model_path=LABEL_NEURAL_NETWORK_MODEL_PATH,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS,
        num_type_of_lines=NUM_TYPE_OF_LINES,
        neural_net_width=NEURAL_NETWORK_WIDTH,
        neural_net_height=NEURAL_NETWORK_WIDTH,
    )

    camera.process(frame_handler=frame_handler)
    camera.release()

    log.info("End working...")
    print('End working...')
