import cv2
import numpy as np
import scipy.misc

from line import *


class Bar:
    """
    Represents bar
    """
    def __init__(self, idx, image):
        self.idx = idx
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.image_original = image
        self.negated_image = self.negate()
        self.vertical_lines = self.detect_vertical_lines()
        self.horizontal_lines = self.detect_horizontal_lines()
        self.lines = self.detect_lines_basic(column=0)
        self.blobs = self.detect_blobs()

    def negate(self):
        processed_bar = cv2.adaptiveThreshold((255 - self.image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 15, -2)
        cv2.imwrite("output/negate.png", processed_bar)
        return processed_bar

    '''
    detects blobs with given parameters.
    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    '''
    def detect_blobs(self):
        # im_with_blobs = self.image.copy()
        # image = cv2.imread('input/notes2.jpg', cv2.IMREAD_GRAYSCALE)
        im_with_blobs = self.image_original.copy()
        params = cv2.SimpleBlobDetector_Params()
        #
        # params.filterByArea = True
        # params.minArea = 20
        #
        # params.filterByCircularity = True
        # params.minCircularity = 0.4
        # params.filterByConvexity = True
        # params.minConvexity = 0.95
        # params.filterByArea = True
        # params.minArea = 20
        #
        params.filterByConvexity = True
        params.minConvexity = 0.9

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(self.image)
        for k in keypoints:
            print(k)
        cv2.drawKeypoints(self.image, keypoints=keypoints, outImage=im_with_blobs, color=(0, 255, 1),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        scipy.misc.imsave('output/outfile_blob.jpg', im_with_blobs)
        return keypoints

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
