import cv2
import numpy as np
from numpy import linalg

from config import *


def get_clef(image, staff):
    """
    Gets the clef from the first staff.

    :param staff: First staff from the image.
    :return:
    """
    width = image.shape[0]

    up = staff.get_lines_locations()[0][0] - WINDOW_WIDTH
    down = staff.get_lines_locations()[0][-1] + WINDOW_WIDTH
    key_width = int((down - up)/1.5)
    i = 0
    while True:
        window = image[up:down, i:i + key_width]
        if window.sum() / window.size < 255 * WHITE_PIXELS_PERCENTAGE:
            break
        if i + key_width > width:
            print("No key detected!")
            break
        i += int(key_width / WINDOW_SHIFT)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("clef.png", window)
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
    return violin_moment, bass_moment


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
    v_moment, b_moment = hu_moments()
    original_clef = get_clef(image, staff)
    original_moment = cv2.HuMoments(cv2.moments(original_clef)).flatten()

    v_moment = log_transform_hu(v_moment)
    b_moment = log_transform_hu(b_moment)
    original_moment = log_transform_hu(original_moment)

    # if VERBOSE:
    #     print(v_moment, b_moment, original_moment)

    if linalg.norm(v_moment - original_moment) - linalg.norm(b_moment - original_moment) > 0.2:
        return "bass"
    else:
        return "violin"

