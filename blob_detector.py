import cv2
from config import *


def detect_blobs(input_image):
    """
    detects blobs with given parameters.
    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    """
    if VERBOSE:
        print("Detecting blobs.")
    im_with_blobs = input_image.copy()

    im_inv = (255 - im_with_blobs)
    kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 500)), shape=cv2.MORPH_RECT)
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = (255 - horizontal_lines)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output_real/8a_lines_vertical.png", horizontal_lines)

    im_with_blobs = horizontal_lines
    im_with_blobs = cv2.cvtColor(im_with_blobs, cv2.COLOR_GRAY2BGR)
    #
    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 20

    params.filterByConvexity = True
    params.minConvexity = 0.9

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(input_image)
    if VERBOSE:
        print("Keypoints length : " + str(len(keypoints)))
    cv2.drawKeypoints(im_with_blobs, keypoints=keypoints, outImage=im_with_blobs, color=(0, 255, 1), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output_real/8b_with_blobs.jpg", im_with_blobs)
    return keypoints
