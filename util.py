import numpy as np


def distance(point1, point2):
    """
    Returns distance between two points in cartesian coordinate system.
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)