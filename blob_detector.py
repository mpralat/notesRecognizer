import cv2

from config import *


def detect_blobs(input_image):
    """
    Detects blobs with given parameters.

    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    """
    if VERBOSE:
        print("Detecting blobs.")
    im_with_blobs = input_image.copy()

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # im_with_blobs = cv2.morphologyEx(im_with_blobs, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("output/10_im_filled_holes.png", im_with_blobs)

    #############################################
    # des = cv2.bitwise_not(im_with_blobs)
    # _, contours, _ = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #
    # contour_list = []
    # for contour in contours:
    #     approx = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
    #     area = cv2.contourArea(contour)
    #     if (len(approx) > 8) & (area > 80) & (area < 200):
    #         contour_list.append(contour)
    #
    # for cnt in contour_list:
    #     cv2.drawContours(des, [cnt], 0, 255, -1)
    #
    # im_with_blobs = cv2.bitwise_not(des)
    # cv2.imwrite("output/10_im_filled_holes.png", im_with_blobs)
    #############################################

    im_inv = (255 - im_with_blobs)
    kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 500)), shape=cv2.MORPH_RECT)
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = (255 - horizontal_lines)

    kernel = cv2.getStructuringElement(ksize=(int(im_inv.shape[1] / 350), 1), shape=cv2.MORPH_RECT)
    vertical_lines = cv2.morphologyEx(255 - horizontal_lines, cv2.MORPH_OPEN, kernel)
    vertical_lines = (255 - vertical_lines)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8a_lines_horizontal_removed.png", horizontal_lines)
        cv2.imwrite("output/8a_lines_vertical_removed.png", vertical_lines)

    im_with_blobs = vertical_lines
    im_with_blobs = cv2.cvtColor(im_with_blobs, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("output/9_im_before_blobbing.png", im_with_blobs)

    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 225
    params.maxArea = 1500
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im_with_blobs)

    if VERBOSE:
        print("Keypoints length : " + str(len(keypoints)))
    cv2.drawKeypoints(im_with_blobs, keypoints=keypoints, outImage=im_with_blobs, color=(0, 255, 1), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8b_with_blobs.jpg", im_with_blobs)
    return keypoints
