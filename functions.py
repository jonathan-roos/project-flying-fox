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

def crop_bat(img, x, y, area):
    x1, x2, y1, y2 = int(x - area), int(x + area), int(y - area), int(y + area)
    
    bat_crop = img[x1: x2, y1: y2]
    cv.imshow("bat", bat_crop)
    cv.waitKey(0)
    
def find_bats(allImgs):
    totalBats = 0
    batDepthMin = 50
    batDepthMax = 400
    bat_location = []
    for img in allImgs:
        blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]  # Find bats using cv.findContours
        bats = []

        # Process the blobs
        for blob in blobs:
            if batDepthMin < cv.contourArea(blob) < batDepthMax:  # Only process blobs with a min / max size
                bats.append(blob)

                rect = cv.minAreaRect(blob)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                # crop bat from image
                # crop_bat(img[0], cx, cy, cv.contourArea(blob))
                # bat_location.append((blob, bat_cords))
                cv.drawContours(img[0], [box], 0, (0, 0, 255), 1)  # Draw bat contours on img

        totalBats += len(bats)
        # cv.imshow("cropped img", img[0])
        # cv.waitKey(0)
    return allImgs, totalBats, bat_location

