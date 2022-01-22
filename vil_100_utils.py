VIL_100_attributes = {
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    13: 10,
}


def get_valid_attribute(attr: int) -> int:
    return VIL_100_attributes.get(attr, attr)
