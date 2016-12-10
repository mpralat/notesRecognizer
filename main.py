from photo_adjuster import adjust_photo
from getting_lines import get_chunks
from blob_detector import detect_blobs
import cv2


def main():
    image = cv2.imread('input/good/easy3.jpg')
    adjusted_photo = adjust_photo(image)
    chunks = get_chunks(adjusted_photo)
    lines = chunks[0].get_lines_locations()
    print(lines)
    blobs = detect_blobs(adjusted_photo)

    # TODO wykrywanie wysokosci nutek


if __name__ == "__main__":
    main()
