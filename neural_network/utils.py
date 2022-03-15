import tensorflow as tf
import numpy as np
import os
from lane_line_recognition.base import draw_polyline_in_frame
from tensorflow.keras.utils import Sequence
from typing import List
from cv2 import imshow as cv2_imshow


class DrawPolylineOnEpochEnd(tf.keras.callbacks.Callback):
    """Draw polyline at the end on the epoch to test the net. It skip one batch on validation but похуй"""

    def __init__(self, test_frame: np.ndarray, train_generator, validation_generator, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.model = model
        self.test_frame = test_frame

    def __filter_coordination_for_resolution(self, polyline: np.ndarray) -> np.ndarray:
        valid = ((polyline[:, 0] > 0) & (polyline[:, 1] > 0)
                 & (polyline[:, 0] < INPUT_SHAPE[0]) & (polyline[:, 1] < INPUT_SHAPE[1]))
        return polyline[valid]

    def __filter_coordinates(self, list_of_polylines: List[np.ndarray]) -> np.ndarray:
        """Remove empty points and coordinates x or y, that is less than 0"""
        list_of_polylines = list(map(lambda x: x.reshape(-1, 2), list_of_polylines))
        return list(map(lambda polyline: self.__filter_coordination_for_resolution(polyline),
                        list_of_polylines))

    def concat_polyline(self, polyline_width: np.ndarray, polyline_height: np.ndarray) -> np.ndarray:
        return np.concatenate((polyline_width.reshape(-1, 1), polyline_height.reshape(-1, 1)), axis=1)

    def test_model(self, generator: Sequence):
        original_frames, original_polylines = generator[0]
        original_frame = original_frames[0]
        draw_original_polylines_frame = draw_polyline_in_frame(
            original_frame,
            original_polylines,
            thickness=2,
            max_lines_per_frame=MAX_LINES_PER_FRAME,
            input_shape=INPUT_SHAPE
        )

        original_frame = np.copy(original_frame)
        frame = np.expand_dims(original_frame, 0)
        res = self.model.predict(frame)

        final_frame = draw_polyline_in_frame(
            draw_original_polylines_frame,
            res,
            thickness=5,
            max_lines_per_frame=MAX_LINES_PER_FRAME,
            input_shape=INPUT_SHAPE
        )
        cv2_imshow(final_frame * 255)

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch + 1}. Draw polylines on train generator...')
        self.test_model(self.train_generator)
        print('Test on test image...')
        self.test_on_image()

    def test_on_image(self):
        frame = np.expand_dims(self.test_frame, 0)
        res = self.model.predict(frame)
        res_frame = draw_polyline_in_frame(
            self.test_frame,
            res, thickness=5,
            max_lines_per_frame=MAX_LINES_PER_FRAME,
            input_shape=INPUT_SHAPE
        )
        cv2_imshow(res_frame * 255)


if __name__ == '__main__':
    from dotenv import load_dotenv
    FINAL_WIDTH = int(os.getenv('FINAL_WIDTH'))
    FINAL_HEIGHT = int(os.getenv('FINAL_HEIGHT'))
    IMAGE_PATH = os.getenv('FRAME_DATASET_PATH')
    JSON_HDF5_DATASET_PATH = os.getenv('JSON_HDF5_DATASET_PATH')
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    INPUT_SHAPE = (FINAL_WIDTH, FINAL_HEIGHT, 1)
