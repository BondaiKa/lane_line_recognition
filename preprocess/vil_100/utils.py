import cv2
import numpy as np
from typing import Tuple, List
from lane_line_recognition.utils import Color

LANE_ID_FULL_LIST = set(range(1, 10))


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
    LineType.SINGLE_WHITE_SOLID: Color.red,
    LineType.SINGLE_WHITE_DOTTED: Color.green,
    LineType.OTHER_LINE: Color.purple,
}

VIL_100_lane_name = {
    LineType.SINGLE_WHITE_SOLID: 'Solid',
    LineType.SINGLE_WHITE_DOTTED: 'Dotted',
    LineType.OTHER_LINE: 'Other',
}


def get_valid_attribute(attr: int) -> int:
    """Change attribute from VIL100_dataset dataset to normal number without missings"""
    _VIL_100_attributes = {
        LineType.NO_LINE: LineType.NO_LINE,
        VIL100Attribute.SINGLE_WHITE_SOLID: LineType.SINGLE_WHITE_SOLID,
        VIL100Attribute.SINGLE_WHITE_DOTTED: LineType.SINGLE_WHITE_DOTTED,
        VIL100Attribute.SINGLE_YELLOW_SOLID: LineType.SINGLE_WHITE_SOLID,
        VIL100Attribute.SINGLE_YELLOW_DOTTED: LineType.SINGLE_WHITE_DOTTED,
        VIL100Attribute.DOUBLE_WHITE_SOLID: LineType.OTHER_LINE,
        VIL100Attribute.DOUBLE_YELLOW_SOLID: LineType.OTHER_LINE,
        VIL100Attribute.DOUBLE_YELLOW_DOTTED: LineType.OTHER_LINE,
        VIL100Attribute.DOUBLE_WHITE_SOLID_DOTTED: LineType.OTHER_LINE,
        VIL100Attribute.DOUBLE_WHITE_DOTTED_SOLID: LineType.OTHER_LINE,
        VIL100Attribute.DOUBLE_SOLID_WHITE_AND_YELLOW: LineType.OTHER_LINE,
    }
    return _VIL_100_attributes.get(attr, LineType.NO_LINE)


def get_colour_from_one_hot_vector(vector: np.ndarray) -> Tuple[int, int, int]:
    """Get colour from one hot vector"""
    #TODO @Karim: fix color. Dash should be green and solid - red
    return VIL_100_colour_line.get(int(np.argmax(vector, axis=0)), VIL_100_colour_line[LineType.OTHER_LINE])
