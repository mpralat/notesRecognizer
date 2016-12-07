import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import *


def draw_lines(hough, image, nlines):
    n_x, n_y = image.shape
    # convert to color image so that you can see the lines
    draw_im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print(hough.shape)
    for ups in hough[:nlines]:
        rho = ups[0][0]
        theta = ups[0][1]
        print("line!")
        x0 = np.cos(theta) * rho
        y0 = np.sin(theta) * rho
        pt1 = (int(x0 + (n_x + n_y) * (-np.sin(theta))),
               int(y0 + (n_x + n_y) * np.cos(theta)))
        pt2 = (int(x0 - (n_x + n_y) * (-np.sin(theta))),
               int(y0 - (n_x + n_y) * np.cos(theta)))
        cv2.line(draw_im, pt1, pt2, (0, 0, 255), 2)

    cv2.imwrite("output_real/1lines.png", draw_im)


def horizontal_lines(result):
    sobeled = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=5)
    sobeled = cv2.erode(sobeled, kernel=np.ones((2, 2), np.uint8), iterations=1)
    # sobeled = cv2.Sobel(sobeled, cv2.CV_64F, 0, 1, ksize=5)
    sobeled = cv2.dilate(sobeled, kernel=np.ones((1, 7), np.uint8), iterations=1)
    # sobeled = cv2.erode(sobeled, kernel=np.ones((1,7), np.uint8), iterations=1)

    cv2.imwrite("output_real/1sobeled.png", sobeled)
    kernel = cv2.getStructuringElement(ksize=(int(sobeled.shape[1] / 60), 1), shape=cv2.MORPH_RECT)
    vertical_lines = cv2.morphologyEx(sobeled, cv2.MORPH_OPEN, kernel)
    # vertical_lines = cv2.dilate(vertical_lines, kernel=(2, 2))

    vertical_lines = cv2.dilate(vertical_lines, kernel=np.ones((1, 7), np.uint8), iterations=1)
    vertical_lines = cv2.Sobel(vertical_lines, cv2.CV_64F, 0, 1, ksize=5)
    vertical_lines = cv2.morphologyEx(vertical_lines, op=cv2.MORPH_CLOSE, kernel=(2, 2))

    cv2.imwrite("output_real/1lines_horizontal2.png", vertical_lines)

    # # vertical_lines = cv2.threshold(vertical_lines, 0, 255, cv2.THRESH_TOZERO_INV)
    # vertical_lines = vertical_lines.astype(dtype=bool).astype(int)
    # print(vertical_lines)
    # skeleton = skeletonize(vertical_lines)
    # cv2.imwrite("skeleton.png", img_as_ubyte(skeleton))

    gray = result.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flag, b = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    cv2.imwrite("output_real/1tresh.jpg", b)

    element = np.ones((3, 3))
    b = cv2.erode(b, element)
    cv2.imwrite("output_real/12erodedtresh.jpg", b)

    edges = cv2.Canny(b, 10, 100, apertureSize=3)
    cv2.imwrite("output_real/13Canny.jpg", edges)

    hough = cv2.HoughLines(edges, 1, np.pi / 150, 200)
    print(len(hough))
    draw_lines(hough, b, 100)
    image = result.copy()
    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                     line_gap=3)
