from bar import *
import documentAdjuster
import getting_lines


def main():
    # bar = Bar(0, image)
    # photo_adjuster.adjust_photo('input/test.jpg')

    image = cv2.imread('input/real4.jpg')
    result = documentAdjuster.main(image)
    # getting_lines.horizontal_lines(result)


if __name__ == "__main__":
    main()
