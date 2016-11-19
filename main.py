from bar import *
import cv2
import scipy.misc


def main():
    image = cv2.imread('input/notes3.jpg')

    bar = Bar(0, image)


if __name__ == "__main__":
    main()
