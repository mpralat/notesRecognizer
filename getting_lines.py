import cv2
import numpy as np

from chunk import Chunk

LINES_DISTANCE_THRESHOLD = 20


def preprocess_image(image):
    sobeled = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobeled = cv2.erode(sobeled, kernel=np.ones((2, 2), np.uint8), iterations=1)
    # sobeled = cv2.Sobel(sobeled, cv2.CV_64F, 0, 1, ksize=5)
    sobeled = cv2.dilate(sobeled, kernel=np.ones((1, 7), np.uint8), iterations=1)
    # sobeled = cv2.erode(sobeled, kernel=np.ones((1,7), np.uint8), iterations=1)

    cv2.imwrite("output_real/4sobeled.png", sobeled)
    kernel = cv2.getStructuringElement(ksize=(int(sobeled.shape[1] / 60), 1), shape=cv2.MORPH_RECT)
    vertical_lines = cv2.morphologyEx(sobeled, cv2.MORPH_OPEN, kernel)
    # vertical_lines = cv2.dilate(vertical_lines, kernel=(2, 2))

    vertical_lines = cv2.dilate(vertical_lines, kernel=np.ones((1, 7), np.uint8), iterations=1)
    vertical_lines = cv2.Sobel(vertical_lines, cv2.CV_64F, 0, 1, ksize=5)
    vertical_lines = cv2.morphologyEx(vertical_lines, op=cv2.MORPH_CLOSE, kernel=(2, 2))

    vertical_lines = (255 - vertical_lines)
    cv2.imwrite("output_real/5lines_horizontal.png", vertical_lines)

    # # vertical_lines = cv2.threshold(vertical_lines, 0, 255, cv2.THRESH_TOZERO_INV)
    # vertical_lines = vertical_lines.astype(dtype=bool).astype(int)
    # print(vertical_lines)
    # skeleton = skeletonize(vertical_lines)
    # cv2.imwrite("skeleton.png", img_as_ubyte(skeleton))

    gray = image.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    element = np.ones((3, 3))
    thresholded = cv2.erode(thresholded, element)

    edges = cv2.Canny(thresholded, 10, 100, apertureSize=3)
    return edges, thresholded


def detect_lines(hough, image, nlines):
    all_lines = set()
    width, height = image.shape
    # convert to color image so that you can see the lines
    lines_image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for result_arr in hough[:nlines]:
        rho = result_arr[0][0]
        theta = result_arr[0][1]
        print("line!")
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        shape_sum = width + height
        x1 = int(x0 + shape_sum * (-b))
        y1 = int(y0 + shape_sum * a)
        x2 = int(x0 - shape_sum * (-b))
        y2 = int(y0 - shape_sum * a)

        # pt1 = (int(x0 + (width + height) * (-np.sin(theta))),
        #        int(y0 + (width + height) * np.cos(theta)))
        # pt2 = (int(x0 - (width + height) * (-np.sin(theta))),
        #        int(y0 - (width + height) * np.cos(theta)))

        start = (x1, y1)
        end = (x2, y2)
        diff = y2 - y1
        if abs(diff) < 10:
            print("difference " + str(diff))
            all_lines.add(int((start[1] + end[1]) / 2))
            cv2.line(lines_image_color, start, end, (0, 0, 255), 2)

    cv2.imwrite("output_real/6lines.png", lines_image_color)
    print(sorted(all_lines))
    print(image.shape)
    return all_lines, lines_image_color


def detect_chunks(all_lines):
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
    for chunk in chunks:
        cv2.line(lines_image_color, (0, chunk[0]), (850, chunk[0]), (0, 255, 255), 2)
        cv2.line(lines_image_color, (0, chunk[1]), (850, chunk[1]), (0, 255, 255), 2)

    cv2.imwrite("output_real/7chunks.png", lines_image_color)


def get_chunks(input_image):
    processed_image, thresholded = preprocess_image(input_image)
    hough = cv2.HoughLines(processed_image, 1, np.pi / 150, 200)
    # print(hough)

    all_lines, lines_image_color = detect_lines(hough, thresholded, 80)
    chunks = detect_chunks(all_lines)
    draw_chunks(lines_image_color, chunks)
    return [Chunk(chunk[0], chunk[1]) for chunk in chunks]
    # image = input_image.copy()
    # edges = canny(image, 2, 1, 25)
    # lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
