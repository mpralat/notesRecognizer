from bar import *
import cv2
import scipy.misc


def main():
    image = cv2.imread('input/notes3.jpg', cv2.IMREAD_GRAYSCALE)
    # edged = cv2.Canny(image2, 1, 200)
    # # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # print(len(cnts))
    # # cv2.drawContours(edged, contours=cnts, contourIdx=-1, color=(0,255,0), thickness=2)
    # scipy.misc.imsave('outfile-ellipses.jpg', edged)
    # cv2.imshow("Closed", edged)
    # cv2.waitKey(0)

    # im_with_blobs = cv2.imread('input/notes2.jpg')
    #
    # params = cv2.SimpleBlobDetector_Params()
    #
    # # # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    # params.filterByArea = True
    # params.minArea = 20
    #
    # params.filterByConvexity = True
    # params.minConvexity = 0.9
    #
    #
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(image)
    # for k in keypoints:
    #     print(k)
    # cv2.drawKeypoints(image, keypoints=keypoints, outImage=im_with_blobs, color=(1, 255, 1),
    #                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # scipy.misc.imsave('outfile.jpg', im_with_blobs)
    # cv2.imwrite("blobs.png", im_with_blobs)
    bar = Bar(0, image)


if __name__ == "__main__":
    main()
