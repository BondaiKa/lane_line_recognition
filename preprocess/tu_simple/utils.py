import cv2
import numpy as np
from lane_line_recognition.utils import Color
from tensorflow.keras.utils import Sequence


class TuSimpleJson:
    frame_path = 'raw_file'
    lane_widths = 'lanes'
    lane_heights = 'h_samples'
