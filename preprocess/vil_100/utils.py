import cv2
import numpy as np
from typing import Tuple, List

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
    POLYLINE_WIDTHS_DATASET_NAME = 'polyline_widths'
    POLYLINE_HEIGHTS_DATASET_NAME = 'polyline_heights'
    LABELS_DATASET_NAME = 'labels'


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
    # todo inital was axis=1
    return VIL_100_colour_line.get(int(np.argmax(vector, axis=0)), VIL_100_colour_line[LineType.OTHER_LINE])


def __filter_coordination_for_resolution(polyline: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    valid = ((polyline[:, 0] > 0) & (polyline[:, 1] > 0)
             & (polyline[:, 0] < input_shape[0]) & (polyline[:, 1] < input_shape[1]))
    return polyline[valid]


def __filter_coordinates(list_of_polylines: List[np.ndarray]) -> np.ndarray:
    """Remove empty points and coordinates x or y, that is less than 0"""
    list_of_polylines = list(map(lambda x: x.reshape(-1, 2), list_of_polylines))
    return list(map(lambda polyline: __filter_coordination_for_resolution(polyline),
                    list_of_polylines))


def concat_polyline(polyline_width: np.ndarray, polyline_height: np.ndarray) -> np.ndarray:
    return np.concatenate((polyline_width.reshape(-1, 1), polyline_height.reshape(-1, 1)), axis=1)


def draw_polyline_in_frame(frame: np.ndarray, polylines: Tuple[np.ndarray, np.ndarray],
                           thickness: int, max_lines_per_frame: int, input_shape: Tuple[int, int, int]) -> np.ndarray:
    copy_frame = np.copy(frame)
    polyline_widths, polyline_heights = polylines[0][0], polylines[1][0]
    polyline_width_list = np.split(polyline_widths, max_lines_per_frame)
    polyline_height_list = np.split(polyline_heights, max_lines_per_frame)

    for polyline_width, polyline_height in zip(polyline_width_list, polyline_height_list):
        polyline_width *= input_shape[0]
        polyline_height *= input_shape[1]
        polyline = concat_polyline(polyline_width, polyline_height)
        polyline = __filter_coordination_for_resolution(polyline, input_shape=input_shape)
        copy_frame = cv2.polylines(copy_frame, np.int32(polyline).reshape((-1, 1, 2)), 1,
                                   color=(255, 0, 0),
                                   thickness=thickness)
    return copy_frame


def test_vil100_generator(generator, input_shape: Tuple[int, int, int],
                          max_lines_per_frame: int, draw_line=False, ) -> None:
    """Test frame and labels on generator"""
    original_frames, original_polylines = generator[0]
    original_frame = original_frames[0]

    draw_original_polylines_frame = draw_polyline_in_frame(
        frame=original_frame,
        polylines=original_polylines,
        max_lines_per_frame=max_lines_per_frame,
        input_shape=input_shape,
        thickness=2
    )
    cv2.imshow(f'frame_with_polyline_{draw_original_polylines_frame.shape}', draw_original_polylines_frame)
    cv2.waitKey(0)