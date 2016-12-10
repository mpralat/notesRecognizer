import cv2
import numpy as np
from config import *


def preprocess_image(image):
    if VERBOSE:
        print("Preprocessing image.")
    gray = image.copy()
    _, thresholded = cv2.threshold(gray, THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)
    element = np.ones((3, 3))
    thresholded = cv2.erode(thresholded, element)
    edges = cv2.Canny(thresholded, 10, 100, apertureSize=3)
    return edges, thresholded


def detect_lines(hough, image, nlines):
    if VERBOSE:
        print("Detecting lines.")
    all_lines = set()
    width, height = image.shape
    # convert to color image so that you can see the lines
    lines_image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for result_arr in hough[:nlines]:
        rho = result_arr[0][0]
        theta = result_arr[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        shape_sum = width + height
        x1 = int(x0 + shape_sum * (-b))
        y1 = int(y0 + shape_sum * a)
        x2 = int(x0 - shape_sum * (-b))
        y2 = int(y0 - shape_sum * a)

        start = (x1, y1)
        end = (x2, y2)
        diff = y2 - y1
        if abs(diff) < LINES_ENDPOINTS_DIFFERENCE:
            all_lines.add(int((start[1] + end[1]) / 2))
            cv2.line(lines_image_color, start, end, (0, 0, 255), 2)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output_real/6lines.png", lines_image_color)

    return all_lines, lines_image_color


def detect_chunks(all_lines):
    if VERBOSE:
        print("Detecting chunks.")
    chunks = []
    lines = []
    all_lines = sorted(all_lines)
    for current_line in all_lines:
        # If current line is far away from last detected line
        if lines and abs(lines[-1] - current_line) > LINES_DISTANCE_THRESHOLD:
            if len(lines) >= 5:
                # Consider it the start of the next chunk.
                # If <5 - not enough lines detected. Probably an anomaly - reject.
                chunks.append((lines[0], lines[-1]))
            lines.clear()
        lines.append(current_line)

    # Process the last line
    if len(lines) >= 5:
        if abs(lines[-2] - lines[-1]) <= LINES_DISTANCE_THRESHOLD:
            chunks.append((lines[0], lines[-1]))
    return chunks


def draw_chunks(lines_image_color, chunks):
    # Draw the chunks
    width = lines_image_color.shape[0]
    for chunk in chunks:
        cv2.line(lines_image_color, (0, chunk[0]), (width, chunk[0]), (0, 255, 255), 2)
        cv2.line(lines_image_color, (0, chunk[1]), (width, chunk[1]), (0, 255, 255), 2)
    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output_real/7chunks.png", lines_image_color)


def get_chunks(input_image):
    processed_image, thresholded = preprocess_image(input_image)
    hough = cv2.HoughLines(processed_image, 1, np.pi / 150, 200)
    all_lines, lines_image_color = detect_lines(hough, thresholded, 80)
    chunks = detect_chunks(all_lines)
    draw_chunks(lines_image_color, chunks)
    return chunks
