import cv2
import numpy as np

from line import *


class Bar:
    """
    Represents bar
    """
    def __init__(self, idx, image):
        self.idx = idx
        self.image = image
        self.negated_image = self.negate()
        self.vertical_lines = self.detect_vertical_lines()
        self.horizontal_lines = self.detect_horizontal_lines()
        self.lines = self.detect_lines_basic(column=0)

    def preprocess(self):
        image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel=(1, 1))
        cv2.imwrite("output/preprocess.png", image)
        return self.image

    def restore_lines(self, array):
        for idx, row in enumerate(array):
            white_count = 0
            for pixel in row:
                if pixel == 255:
                    white_count += 1
            if white_count > (3/4) * len(row):
                array[idx] = 255

    def remove_lines_noise(self, array):
        height, width = array.shape
        array_copy = array.copy()
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if (array_copy[row][col - 1] == 0 and array_copy[row][col + 1] == 0) and not (array_copy[row - 1][col] == 255 and array_copy[row + 1][col] == 255):
                    array[row][col] = 0
                    if array_copy[row][col] != array[row][col]:
                        print(row, col)

    def restore_notes(self, array):
        height, width = array.shape
        array_copy = array.copy()
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if array_copy[row - 1][col] == 255 and array_copy[row + 1][col] == 255:
                    array[row][col] = 255
                    if array_copy[row][col] != array[row][col]:
                        print(row, col)

    def negate(self):
        processed_bar = cv2.adaptiveThreshold((255 - self.image), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -30)
        # _, processed_bar = cv2.threshold((255 - self.image), 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TOZERO)
        self.restore_lines(processed_bar)
        self.remove_lines_noise(processed_bar)
        self.restore_notes(processed_bar)
        cv2.imwrite("output/negate.png", processed_bar)
        return processed_bar

    def detect_horizontal_lines(self):
        kernel = cv2.getStructuringElement(ksize=(int(self.negated_image.shape[1] / 4), 1), shape=cv2.MORPH_RECT)
        vertical_lines = cv2.morphologyEx(self.negated_image, cv2.MORPH_OPEN, kernel)
        # vertical_lines = cv2.dilate(vertical_lines, kernel=(2, 2))
        cv2.imwrite("output/lines_horizontal.png", vertical_lines)
        return vertical_lines

    def detect_vertical_lines(self):
        kernel = cv2.getStructuringElement(ksize=(1, int(self.negated_image.shape[0] / 15)), shape=cv2.MORPH_RECT)
        horizontal_lines = cv2.morphologyEx(self.negated_image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("output/lines_vertical.png", horizontal_lines)
        return horizontal_lines

    def detect_lines(self):
        edges = cv2.Canny(self.horizontal_lines, 130, 200)
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=15, maxLineGap=10)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        detected_lines = [Line(idx=i, y_pos=x[0][1]) for i, x in enumerate(lines)]
        lines_img = np.zeros_like(self.negated_image)
        for i, line in enumerate(lines):
            for x1, y1, x2, y2 in line:
                cv2.line(lines_img, (x1,y1),(x2,y2), (255,0,0), 2)
                cv2.imwrite("output/lines" + str(i) + ".png", lines_img)

        return detected_lines

    def detect_lines_basic(self, column):
        while True:
            basic_lines = []
            i = 10
            for row in range (0, self.horizontal_lines.shape[0]-1):
                value = self.horizontal_lines[row][column]
                if value != self.horizontal_lines[row+1][column]:
                    if i % 2 == 0:
                        y_begin = row
                    else:
                        basic_lines.append(Line(i//2 + 1, y_begin, row))
                    i -= 1
            if len(basic_lines) == 5:
                break
            else:
                column += 1
        print("column: " + str(column))
        for x in basic_lines:
            print(x.idx, x.y_begin, x.y_end)
        return basic_lines




        # lines_img = np.zeros_like(self.processed_bar)
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(lines_img, (x1,y1),(x2,y2), (255,0,0), 2)
        #
        # cv2.imwrite("lines.png", lines_img)


        # def detect_ellipses(self):
        #     output = (self.image).copy()
        #     edges = canny(self.image, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
        #     result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=3, max_size=5)
        #     result.sort(order='accumulator')
        #
        #     for ellipse in result:
        #         yc, xc, a, b = [int(round(x)) for x in ellipse[1:5]]
        #         orientation = ellipse[5]
        #         cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        #         # output[cy,cx] = (255, 0,0)
