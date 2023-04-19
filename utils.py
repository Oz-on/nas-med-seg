"""
Author: Oskar Domingos
"""


def constrain(min_v: int, max_v: int, value: int) -> int:
    if value < min_v:
        return min_v

    elif value > max_v:
        return max_v

    return value
