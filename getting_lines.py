import cv2
import numpy as np

LINES_DISTANCE_THRESHOLD = 20


def preprocess_image(image):
    sobeled = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
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

    gray = image.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flag, thresholded = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    cv2.imwrite("output_real/1tresh.jpg", thresholded)

    element = np.ones((3, 3))
    thresholded = cv2.erode(thresholded, element)
    cv2.imwrite("output_real/12erodedtresh.jpg", thresholded)

    edges = cv2.Canny(thresholded, 10, 100, apertureSize=3)
    cv2.imwrite("output_real/13Canny.jpg", edges)
    return edges, thresholded


def detect_lines(hough, image, nlines):
    all_lines = set()
    n_x, n_y = image.shape
    # convert to color image so that you can see the lines
    lines_image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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

        all_lines.add(int((pt1[1] + pt2[1]) / 2))
        cv2.line(lines_image_color, pt1, pt2, (0, 0, 255), 2)
    cv2.imwrite("output_real/1lines.png", lines_image_color)
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

    cv2.imwrite("output_real/1chunks.png", lines_image_color)


def get_chunks(input_image):
    processed_image, b = preprocess_image(input_image)
    hough = cv2.HoughLines(processed_image, 1, np.pi / 150, 200)

    all_lines, lines_image_color = detect_lines(hough, b, 100)
    chunks = detect_chunks(all_lines)
    draw_chunks(lines_image_color, chunks)
    return chunks
    # image = input_image.copy()
    # edges = canny(image, 2, 1, 25)
    # lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
