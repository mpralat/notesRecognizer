import cv2

from blob_detector import detect_blobs
from getting_lines import get_chunks
from photo_adjuster import adjust_photo


def main():
    image = cv2.imread('input/good/medium4.jpg')
    adjusted_photo = adjust_photo(image)
    chunks = get_chunks(adjusted_photo)
    lines = chunks[0].get_lines_locations()
    print(lines)
    blobs = detect_blobs(adjusted_photo)

    # TODO Zmienic detect_chunks zeby rysowala do konca obrazka, a nie do 800 piksela
    # TODO wykrywanie wysokosci nutek


if __name__ == "__main__":
    main()
