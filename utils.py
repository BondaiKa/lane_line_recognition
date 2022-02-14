from typing import List, Tuple
import numpy as np
import cv2


def one_hot_list_encoder(target_class_idx: int, num_classes: int) -> np.ndarray:
    """One-hot list encoder"""
    target_vector = np.zeros(num_classes)
    target_vector[target_class_idx] = 1
    return target_vector


def test_generator(generator, draw_line=False) -> None:
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


def test_model(model, generator) -> None:
    def filter_coordination_for_resolution(polyline: np.ndarray) -> np.ndarray:
        valid = ((polyline[:, 0] > 0) & (polyline[:, 1] > 0)
                 & (polyline[:, 0] < 1280) & (polyline[:, 1] < 960))
        return polyline[valid]

    def filter_coordinates(list_of_polylines: List[np.ndarray]) -> np.ndarray:
        """Remove empty points and coordinates x or y, that is less than 0"""
        list_of_polylines = list(map(lambda x: x.reshape(-1, 2), list_of_polylines))
        return list(map(lambda polyline: filter_coordination_for_resolution(polyline),
                        list_of_polylines))

    original_frames, original_labels = generator[0]
    original_frame = original_frames[0]

    print(original_frame.shape)
    original_frame = np.copy(original_frame)
    frame = np.expand_dims(original_frame, 0)
    res = model.predict(frame.T)

    polylines, label_1, label_2 = res[0], res[1], res[2]
    polylines = np.hsplit(polylines, 2)
    polylines = filter_coordinates(polylines)

    # TODO @Karim: use code to show color according to results
    for points, color in zip(polylines, [(255, 0, 255), (0, 255, 255)]):
        original_frame = cv2.polylines(original_frame, np.int32(points).reshape((-1, 1, 2)), 1, color=color,
                                       thickness=5)

    cv2.imshow(original_frame * 255)
    cv2.waitKey(0)


# TODO @Karim: remove after debugging perspective_transformation
def draw_sequence_in_img(frame: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = thickness = 3
    lineType = 2
    frame_points = frame.copy()
    for num, point in enumerate(points):
        frame_points = cv2.putText(frame_points, str(num + 1), point, font, fontScale, color, thickness,
                                   lineType)
    cv2.imshow(f"Frame_with_points_{color[0]}_{color[1]}_{color[2]}", frame_points)
