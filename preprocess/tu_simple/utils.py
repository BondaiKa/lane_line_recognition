import cv2
import numpy as np
from lane_line_recognition.utils import Color
from tensorflow.keras.utils import Sequence


class TuSimpleJson:
    frame_path = 'raw_file'
    lane_widths = 'lanes'
    lane_heights = 'h_samples'


class TuSimpleHdf5:
    group_name = 'polylines'
    dataset_polylines_width = 'polylines_width'
    dataset_polylines_height = 'polylines_height'


def concat_polyline(polyline_width: np.ndarray, polyline_height: np.ndarray) -> np.ndarray:
    return np.concatenate((polyline_width.reshape(-1, 1), polyline_height.reshape(-1, 1)), axis=1)


def test_tu_simple_generator(generator: Sequence, draw_line=False) -> None:
    """Test tu simple generator"""
    original_frames, original_polylines = generator[0]
    original_frame = original_frames[0]
    polyline_widths, polyline_height = original_polylines[0][0], original_polylines[1][0]

    polyline_width_list = np.split(polyline_widths, 5)

    if draw_line:
        for polyline_width, color in zip(polyline_width_list, Color.sequence[:len(polyline_width_list)]):
            polyline = concat_polyline(polyline_width, polyline_height)
            polyline /= 0.003
            original_frame = cv2.polylines(original_frame, np.int32(polyline).reshape((-1, 1, 2)), 1,
                                           color=color,
                                           thickness=5)

    cv2.imshow(f'frame_with_polyline_{original_frame.shape}', original_frame)
    cv2.waitKey(0)
    print(original_frame.shape)
