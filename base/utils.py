from typing import Tuple, List
import numpy as np
import cv2


class LaneLineRecognitionHDF5:
    root_folder = 'hdf5'
    group_name = 'frame_polylines_labels'
    polyline_widths_dataset_name = 'polyline_widths'
    polyline_heights_dataset_name = 'polyline_heights'
    labels_dataset_name = 'labels'


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


def test_generator(generator, input_shape: Tuple[int, int, int],
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
