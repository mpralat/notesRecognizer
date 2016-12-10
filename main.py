from photo_adjuster import adjust_photo
from getting_lines import get_chunks
import cv2


def main():
    image = cv2.imread('input/real4.jpg')
    adjusted_photo = adjust_photo(image)
    chunks = get_chunks(adjusted_photo)

    # TODO Zmienic funkcje get_chunks zeby zwracala liste klas Chunk, zamiast tupli z wartosia min i max
    # TODO Zmienic detect_chunks zeby rysowala do konca obrazka, a nie do 800 piksela
    # TODO photo_adjuster dla innych zdjec
    # TODO wykrywanie wysokosci nutek


if __name__ == "__main__":
    main()
