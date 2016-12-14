from blob_detector import detect_blobs
from getting_lines import get_staffs
from note import *
from photo_adjuster import adjust_photo
import argparse
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default='input/good/dark2.jpg'
    )

    return parser.parse_args()


def main():
    args = parse()
    image = cv2.imread(args.input)
    adjusted_photo = adjust_photo(image)
    staffs = get_staffs(adjusted_photo)
    blobs = detect_blobs(adjusted_photo, staffs)
    notes = extract_notes(blobs, staffs, adjusted_photo)
    draw_notes_pitch(adjusted_photo, notes)


if __name__ == "__main__":
    main()
