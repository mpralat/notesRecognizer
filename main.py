from photo_adjuster import adjust_photo
from getting_lines import get_chunks
import cv2
from blob_detector import detect_blobs

def main():
    image = cv2.imread('input/good/medium4.jpg')
    adjusted_photo = adjust_photo(image)
    chunks = get_chunks(adjusted_photo)
    blobs = detect_blobs(adjusted_photo)

    # TODO Zmienic funkcje get_chunks zeby zwracala liste klas Chunk, zamiast tupli z wartosia min i max

    # TODO Zmienic detect_chunks zeby rysowala do konca obrazka, a nie do 800 piksela
    # TODO photo_adjuster dla innych zdjec -DONE? (popracowac nad tymi trudnymi mozna potem)
    # TODO wykrywanie wysokosci nutek


if __name__ == "__main__":
    main()
