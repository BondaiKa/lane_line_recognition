from abc import ABC
import numpy as np
import cv2


class FrameHandler:
    """Abstract frame handler"""

    @staticmethod
    def add_lane_markup(img: np.ndarray, polylines: np.ndarray) -> np.ndarray:
        """get an image and polylines and return markup image"""
        return np.apply_along_axis(cv2.polylines, axis=1, arr=polylines,
                                   img=img, isClosed=True, color=(255, 0, 0), thickness=2)
