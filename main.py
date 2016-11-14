import cv2
from skimage.transform import hough_ellipse
from skimage.feature import canny
from skimage.draw import ellipse_perimeter


class Bar:
    """
    Represents bar
    """
    def __init__(self, idx, image):
        self.idx = idx
        self.image = image

    def detect_ellipses(self):
        output = (self.image).copy()
        edges = canny(self.image, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
        result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=3, max_size=5)
        result.sort(order='accumulator')

        for ellipse in result:
            yc, xc, a, b = [int(round(x)) for x in ellipse[1:5]]
            orientation = ellipse[5]
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            # output[cy,cx] = (255, 0,0)



        print("boze jaki dramat")

        cv2.imwrite("out.png", output)


def main():
    image = cv2.imread('input/notes1.jpg', cv2.IMREAD_GRAYSCALE)
    bar = Bar(0, image)
    bar.detect_ellipses()


if __name__ == "__main__":
    main()
