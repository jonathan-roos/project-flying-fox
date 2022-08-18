import cv2 as cv
import numpy as np


def chop_img(img, step):
    allImgs = []
    width = 640
    height = 512
    i, j = 0, 0
    for i in range(step):
        for j in range(step):
            # crop the image into step*rows and step*columns
            imgCrop = img[int(0 + height / step * i): int(height / step + height / step * i),
                      int(0 + width / step * j): int(width / step + width / step * j)]
            imgResize = cv.resize(imgCrop, (640, 512))  # Resize image
            imgAug = augment(imgResize)
            allImgs.append((imgResize, imgAug))
            j += 1
        i += 1
    return allImgs

def augment(img):
    kernel = np.ones((5, 5), np.uint8)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 0)  # apply gaussian blur
    imgThresh = cv.adaptiveThreshold(  # apply adaptive threshold
    imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 5)
    imgDilation = cv.dilate(imgThresh, kernel, iterations=1)  # apply dilation to amplify threshold result
    return imgDilation

def find_bats(allImgs):
    totalBats = 0
    batDepthMin = 50
    batDepthMax = 400
    bat_location = []
    for img in allImgs:
        blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]  # Find bats using cv.findContours

        # Bat depth!

        bats = []

        # Process the blobs
        for blob in blobs:
            if batDepthMin < cv.contourArea(blob) < batDepthMax:  # Only process blobs with a min / max size
                bats.append(blob)
                # Circle drawing maths shit
                M = cv.moments(blob)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bat_cords = (cx, cy)
                bat_location.append((blob, bat_cords))

        totalBats += len(bats)
        cv.drawContours(img[0], bats, -1, (0, 255, 255), 1)  # Draw bat contours on img
    return allImgs, totalBats, bat_location