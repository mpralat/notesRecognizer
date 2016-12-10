import cv2
import scipy.misc


def detect_blobs(input_image):
    """
    detects blobs with given parameters.
    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    """
    # im_with_blobs = self.image.copy()
    # image = cv2.imread('input/notes2.jpg', cv2.IMREAD_GRAYSCALE)
    im_with_blobs = input_image.copy()

    im_inv = (255 - im_with_blobs)

    kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 500)), shape=cv2.MORPH_RECT)
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = (255 - horizontal_lines)

    cv2.imwrite("output_real/lines_vertical.png", horizontal_lines)


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
    print(len(keypoints))
    cv2.drawKeypoints(im_with_blobs, keypoints=keypoints, outImage=im_with_blobs, color=(0, 255, 1),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    scipy.misc.imsave('output_real/outfile_blob_3.jpg', im_with_blobs)
    cv2.imwrite("output_real/with_blobs.jpg", im_with_blobs)
    return keypoints
