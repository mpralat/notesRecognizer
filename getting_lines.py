import cv2
import numpy as np

def horizontal_lines(result):
    sobeled = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=5)
    sobeled = cv2.erode(sobeled, kernel=np.ones((2, 2), np.uint8), iterations=1)
    # sobeled = cv2.Sobel(sobeled, cv2.CV_64F, 0, 1, ksize=5)
    sobeled = cv2.dilate(sobeled, kernel=np.ones((1, 7), np.uint8), iterations=1)
    # sobeled = cv2.erode(sobeled, kernel=np.ones((1,7), np.uint8), iterations=1)

    cv2.imwrite("sobeled.png", sobeled)
    kernel = cv2.getStructuringElement(ksize=(int(sobeled.shape[1] / 60), 1), shape=cv2.MORPH_RECT)
    vertical_lines = cv2.morphologyEx(sobeled, cv2.MORPH_OPEN, kernel)
    # vertical_lines = cv2.dilate(vertical_lines, kernel=(2, 2))


    array = vertical_lines
    for i in range(0, 4):
        for idx, row in enumerate(array):
            white_count = 0
            for pixel in row:
                if pixel == 255:
                    white_count += 1
            if white_count > (5 / 4) * len(row):
                array[idx] = 255

    height, width = array.shape
    array_copy = array.copy()
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if (array_copy[row][col - 1] == 0 and array_copy[row][col + 1] == 0) \
                    and not (array_copy[row - 1][col] == 255 and array_copy[row + 1][col] == 255):
                array[row][col] = 0
                if array_copy[row][col] != array[row][col]:
                    print(row, col)

    vertical_lines = cv2.dilate(array, kernel=np.ones((1, 7), np.uint8), iterations=1)
    vertical_lines = cv2.Sobel(vertical_lines, cv2.CV_64F, 0, 1, ksize=5)
    cv2.imwrite("lines_horizontal2.png", vertical_lines)
