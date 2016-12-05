from bar import *
import documentAdjuster
import getting_lines

def main():
    image = cv2.imread('input/notes3.jpg')
    # bar = Bar(0, image)
    # photo_adjuster.adjust_photo('input/test.jpg')
    img = cv2.imread('out_test/Otsus.jpg')
    result = documentAdjuster.main()
    getting_lines.horizontal_lines(result)

if __name__ == "__main__":
    main()
