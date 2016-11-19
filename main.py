from bar import *
import cv2


def main():
    image = cv2.imread('input/notes1.jpg')
    bar = Bar(0, image)


if __name__ == "__main__":
    main()
