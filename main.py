from bar import *


def main():
    image = cv2.imread('input/notes1.jpg', cv2.IMREAD_GRAYSCALE)
    bar = Bar(0, image)


if __name__ == "__main__":
    main()
