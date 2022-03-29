import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import Sequence
from typing import List, Tuple
from lane_line_recognition.base import draw_polyline_in_frame
from google.colab.patches import cv2_imshow

class DrawPolylineOnEpochEnd(tf.keras.callbacks.Callback):
    """Draw polyline at the end on the epoch to test the net. It skip one batch on validation but похуй"""

    def __init__(self, test_frame_paths: List[str], train_generator,
                 validation_generator, model, input_shape: Tuple[int, int, int],
                 target_shape: Tuple[int, int], max_lines_per_frame: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.model = model
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.max_lines_per_frame = max_lines_per_frame

        frames = []
        for frame_path in test_frame_paths:
            frame = tf.keras.utils.load_img(frame_path,
                                            color_mode='grayscale',
                                            target_size=(self.target_shape[1], self.target_shape[0])
                                            )
            frame = tf.keras.preprocessing.image.img_to_array(frame)
            frame = frame * 1 / 255.
            frames.append(frame)

        self.test_frames = np.array(frames)

    def __filter_coordination_for_resolution(self, polyline: np.ndarray) -> np.ndarray:
        valid = ((polyline[:, 0] > 0) & (polyline[:, 1] > 0)
                 & (polyline[:, 0] < self.input_shape[0]) & (polyline[:, 1] < self.input_shape[1]))
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
            max_lines_per_frame=self.max_lines_per_frame,
            input_shape=self.input_shape
        )

        original_frame = np.copy(original_frame)
        frame = np.expand_dims(original_frame, 0)
        res = self.model.predict(frame)

        final_frame = draw_polyline_in_frame(
            draw_original_polylines_frame,
            res,
            thickness=5,
            max_lines_per_frame=self.max_lines_per_frame,
            input_shape=self.input_shape
        )
        cv2_imshow(final_frame * 255)

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch + 1}. Draw polylines on train generator...')
        self.test_model(self.train_generator)
        print('Test on test image...')
        self.test_on_image()

    def test_on_image(self):
        for frame in self.test_frames:
            frame = np.copy(frame)
            predict_frame = np.expand_dims(frame, 0)
            res = self.model.predict(predict_frame)
            res_frame = draw_polyline_in_frame(
                frame,
                res, thickness=5,
                max_lines_per_frame=self.max_lines_per_frame,
                input_shape=self.input_shape,
            )
            cv2_imshow(res_frame * 255)
