import cv2
import numpy as np
from numpy import linalg

from config import *


def get_clef(image, staff):
    """
    Gets the clef from the first staff.

    :param image: image to get the clef from
    :param staff: First staff from the image.
    :return:
    """
    i = 0
    width = image.shape[0]

    window_width = int(2 / 5 * (staff.max_range - staff.min_range))

    up = staff.lines_location[0] - window_width
    down = staff.lines_location[-1] + window_width
    key_width = int((down - up) / 1.3)
    while True:
        window = image[up:down, i:i + key_width]
        if window.sum() / window.size < int(255 * WHITE_PIXELS_PERCENTAGE):
            break
        if i + key_width > width:
            print("No key detected!")
            break
        i += int(key_width / WINDOW_SHIFT)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/7clef.png", window)
    return window


def hu_moments():
    """
    Computes hu moments for sample violin and bass keys.

    :return: violin and bass clef hu moments
    """
    violin_key = cv2.imread("clef_samples/violin_clef.png", 0)
    bass_key = cv2.imread("clef_samples/bass_clef2.png", 0)
    violin_moment = cv2.HuMoments(cv2.moments(violin_key)).flatten()
    bass_moment = cv2.HuMoments(cv2.moments(bass_key)).flatten()
    return log_transform_hu(violin_moment), log_transform_hu(bass_moment)


def log_transform_hu(hu_moment):
    """
    Transforms hu moments so they can be easily compared.
    """
    return -np.sign(hu_moment) * np.log10(np.abs(hu_moment))


def classify_clef(image, staff):
    """
    Uses Hu moments to classify the clef - violin or bass.

    :return: A string indicating the clef
    """
    original_clef = get_clef(image, staff)

    v_moment, b_moment = hu_moments()
    v_moment = v_moment[:3]
    b_moment = b_moment[:3]

    original_moment = cv2.HuMoments(cv2.moments(original_clef)).flatten()
    original_moment = log_transform_hu(original_moment)
    original_moment = original_moment[:3]

    # v_difference = sum([j - i for i, j in zip(v_moment, original_moment)])
    # b_difference = sum([j - i for i, j in zip(b_moment, original_moment)])
    # print("Hu moments differences: " + str(v_difference) + ' ' + str(b_difference))
    if linalg.norm(v_moment - original_moment) < linalg.norm(b_moment - original_moment):
        return "violin"
    else:
        return "bass"
