import cv2
from config import *
import numpy as np
from numpy import linalg


def get_key(image, chunk):
    """
    Gets the key from the first chunk.
    :param chunk: First chunk from the image.
    :return:
    """
    width = image.shape[0]

    up = chunk.get_lines_locations()[0] - WINDOW_WIDTH
    down = chunk.get_lines_locations()[-1] + WINDOW_WIDTH
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
        cv2.imwrite("key.png", window)
    return window


def hu_moments():
    """
    Computes hu moments for sample violin and bass keys.
    :return: violin and bass keys hu moments
    """

    violin_key = cv2.imread("key_samples/violin_key.png", 0)
    bass_key = cv2.imread("key_samples/bass_key2.png", 0)
    violin_moment = cv2.HuMoments(cv2.moments(violin_key)).flatten()
    bass_moment = cv2.HuMoments(cv2.moments(bass_key)).flatten()

    return violin_moment, bass_moment


def log_transform_hu(hu_moment):
    """
    Transforms hu moments so they can be easily compared.
    """
    return -np.sign(hu_moment) * np.log10(np.abs(hu_moment))


def classify_key(image, chunk):
    """
    Uses Hu moments to classify the clef - violin or bass.
    :return: A string indicating the key
    """
    v_moment, b_moment = hu_moments()
    original_key = get_key(image, chunk)
    original_moment = cv2.HuMoments(cv2.moments(original_key)).flatten()

    v_moment = log_transform_hu(v_moment)
    b_moment = log_transform_hu(b_moment)
    original_moment = log_transform_hu(original_moment)

    print(v_moment, b_moment, original_moment)
    if linalg.norm(v_moment - original_moment) - linalg.norm(b_moment - original_moment) > 0.2:
        return "bass"
    else:
        return "violin"

