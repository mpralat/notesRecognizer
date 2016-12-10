import sys

import cv2
import numpy as np

from config import *


def distance(point1, point2):
    """
    Returns distance between two points in cartesian coordinate system.
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def adjust_photo(image):
    """
    Detects a sheet of paper in the image. Crops, rotates and performs thresholding on it.

    :param image: image to adjust
    :return: adjusted photo prepared for further analysis
    """
    if VERBOSE:
        print("Adjusting photo.")
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    edged = cv2.Canny(blur, 0, 50)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/1canny.jpg", edged)

    _, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        # Douglas Pecker algorithm - reduces the number of points in a curve
        epsilon = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * epsilon, True)
        if len(approx) == 4:
            sheet = approx
            break

    if 'sheet' not in locals():
        print("Couldn't find a paper sheet in the picture!")
        sys.exit()

    approx = np.asarray([x[0] for x in sheet.astype(dtype=np.float32)])

    # top_left has the smallest sum, bottom_right has the biggest
    top_left = min(approx, key=lambda t: t[0] + t[1])
    bottom_right = max(approx, key=lambda t: t[0] + t[1])
    top_right = max(approx, key=lambda t: t[0] - t[1])
    bottom_left = min(approx, key=lambda t: t[0] - t[1])

    max_width = int(max(distance(bottom_right, bottom_left), distance(top_right, top_left)))
    max_height = int(max(distance(top_right, bottom_right), distance(top_left, bottom_left)))

    arr = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    rectangle = np.asarray([top_left, top_right, bottom_right, bottom_left])

    m = cv2.getPerspectiveTransform(rectangle, arr)
    dst = cv2.warpPerspective(image, m, (max_width, max_height))

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("output/2with_contours.png", image)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    _, result = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/3adjusted_photo.png", result)
    return result
