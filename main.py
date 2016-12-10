import cv2

from blob_detector import detect_blobs
from getting_lines import get_staffs
from photo_adjuster import adjust_photo
from hu import classify_key

def main():
    image = cv2.imread('input/good/easy1.jpg')
    adjusted_photo = adjust_photo(image)
    staffs = get_staffs(adjusted_photo)
    lines = staffs[0].get_lines_locations()
    blobs = detect_blobs(adjusted_photo)
    classify_key(adjusted_photo, staffs[0])

    # TODO wykrywanie wysokosci nutek


if __name__ == "__main__":
    main()
