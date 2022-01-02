from typing import List


def one_hot_list_encoder(target_class_idx, num_classes) -> List[int]:
    """one-hot list encoder"""
    target_vector = [0] * num_classes
    target_vector[target_class_idx] = 1
    return target_vector
