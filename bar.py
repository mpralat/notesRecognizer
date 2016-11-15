import cv2


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

    def negate(self):
        processed_bar = cv2.adaptiveThreshold((255-self.image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
        cv2.imwrite("output/negate.png", processed_bar)
        return processed_bar

    def detect_horizontal_lines(self):
        kernel = cv2.getStructuringElement(ksize=(int(self.negated_image.shape[1] / 10), 1), shape=cv2.MORPH_RECT)
        vertical_lines = cv2.morphologyEx(self.negated_image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("output/lines_horizontal.png", vertical_lines)
        return vertical_lines

    def detect_vertical_lines(self):
        kernel = cv2.getStructuringElement(ksize=(1, int(self.negated_image.shape[0] / 15)), shape=cv2.MORPH_RECT)
        horizontal_lines = cv2.morphologyEx(self.negated_image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("output/lines_vertical.png", horizontal_lines)
        return horizontal_lines

    # def detect_lines(self):
    #     edges = cv2.Canny(self.processed_bar, 5, 200)
    #     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=1, maxLineGap=10)
    #     lines_img = np.zeros_like(self.processed_bar)
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             cv2.line(lines_img, (x1,y1),(x2,y2), (255,0,0), 2)
    #
    #     cv2.imwrite("lines.png", lines_img)


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