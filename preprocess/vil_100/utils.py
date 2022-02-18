import cv2
import numpy as np
from typing import Tuple

LANE_ID_FULL_LIST = set(range(1, 3))


class Vil100Json:
    ANNOTATIONS = 'annotations'
    # Annotations
    ATTRIBUTE = 'attribute'
    LANE = 'lane'
    LANE_ID = 'lane_id'
    POINTS = 'points'

    INFO = 'info'
    # Info
    IMAGE_PATH = 'image_path'
    HEIGHT = 'height'
    WIDTH = 'width'


class VIL100Attribute:
    """Lane Attribute id (type lane) in jsons"""
    SINGLE_WHITE_SOLID = 1
    SINGLE_WHITE_DOTTED = 2
    SINGLE_YELLOW_SOLID = 3
    SINGLE_YELLOW_DOTTED = 4
    DOUBLE_WHITE_SOLID = 5
    DOUBLE_YELLOW_SOLID = 7
    DOUBLE_YELLOW_DOTTED = 8
    DOUBLE_WHITE_SOLID_DOTTED = 9
    DOUBLE_WHITE_DOTTED_SOLID = 10
    DOUBLE_SOLID_WHITE_AND_YELLOW = 13


class LineType:
    """Type lane in our task"""
    NO_LINE = 0
    SINGLE_WHITE_SOLID = 1
    SINGLE_WHITE_DOTTED = 2
    OTHER_LINE = 3

    ALL_LINES = {NO_LINE, SINGLE_WHITE_SOLID, SINGLE_WHITE_DOTTED, OTHER_LINE}


VIL_100_colour_line = {
    LineType.SINGLE_WHITE_SOLID: (255, 0, 0),
    LineType.SINGLE_WHITE_DOTTED: (0, 255, 0),
    LineType.OTHER_LINE: (0, 0, 255),
    # 4: (255, 255, 0),  # single yellow dotted
    # 5: (255, 0, 0),  # double white solid
    # 6: (255, 125, 0),  # double yellow solid
    # 7: (255, 255, 0),  # double yellow dotted
    # 8: (255, 0, 0),  # double white solid dotted
    # 9: (255, 0, 0),  # double white dotted solid
    # 10: (255, 0, 0),  # double solid white and yellow
}


class VIL100HDF5:
    ROOT_FOLDER = 'hdf5'
    GROUP_NAME = 'frame_polylines_labels'
    POLYLINES_DATASET_NAME = 'polylines'
    LABELS_DATASET_NAME = 'labels'


def get_valid_attribute(attr: int) -> int:
    """Change attribute from VIL100_dataset dataset to normal number without missings"""
    _VIL_100_attributes = {
        LineType.NO_LINE: LineType.NO_LINE,
        VIL100Attribute.SINGLE_WHITE_SOLID: LineType.SINGLE_WHITE_SOLID,
        VIL100Attribute.SINGLE_WHITE_DOTTED: LineType.SINGLE_WHITE_DOTTED,
        VIL100Attribute.SINGLE_YELLOW_SOLID: LineType.SINGLE_WHITE_SOLID,
        VIL100Attribute.SINGLE_YELLOW_DOTTED: LineType.SINGLE_WHITE_DOTTED,
        VIL100Attribute.DOUBLE_WHITE_SOLID: LineType.DOUBLE_WHITE_SOLID,
        VIL100Attribute.DOUBLE_YELLOW_SOLID: LineType.DOUBLE_WHITE_SOLID,
        VIL100Attribute.DOUBLE_YELLOW_DOTTED: LineType.SINGLE_WHITE_DOTTED,
        VIL100Attribute.DOUBLE_WHITE_SOLID_DOTTED: LineType.SINGLE_WHITE_DOTTED,
        VIL100Attribute.DOUBLE_WHITE_DOTTED_SOLID: LineType.SINGLE_WHITE_DOTTED,
        VIL100Attribute.DOUBLE_SOLID_WHITE_AND_YELLOW: LineType.DOUBLE_WHITE_SOLID,
    }
    return _VIL_100_attributes.get(attr, LineType.NO_LINE)


def get_colour_from_one_hot_vector(vector: np.ndarray) -> Tuple[int, int, int]:
    """Get colour from one hot vector"""
    # todo inital was axis=1
    return VIL_100_colour_line.get(int(np.argmax(vector, axis=0)), VIL_100_colour_line[LineType.OTHER_LINE])


def test_vil100_generator(generator, draw_line=False) -> None:
    """Test frame and labels on generator"""
    original_frames, original_labels = generator[0]
    original_frame = original_frames[0]
    polyline_1, polyline_2, original_label_1, original_label_2 = original_labels[0][0], original_labels[1][0], \
                                                                 original_labels[2][0], original_labels[3][0]

    if draw_line:
        original_frame = cv2.polylines(original_frame, np.int32(polyline_1).reshape((-1, 1, 2)), 1, color=(255, 0, 255),
                                       thickness=5)
        original_frame = cv2.polylines(original_frame, np.int32(polyline_2).reshape((-1, 1, 2)), 1, color=(0, 255, 0),
                                       thickness=5)
    cv2.imshow(f'frame_with_polyline_{original_frame.shape}', original_frame)
    cv2.waitKey(0)
    print(original_frame.shape)
