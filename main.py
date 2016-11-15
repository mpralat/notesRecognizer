from bar import *
import note


def main():
    image = cv2.imread('input/notes2.jpg', cv2.IMREAD_GRAYSCALE)
    bar = Bar(0, image)


if __name__ == "__main__":
    main()
