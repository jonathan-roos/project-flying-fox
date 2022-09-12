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

def crop_bat(img, center_x, center_y):
    area = 25
    x1, x2, y1, y2 = int(center_x - area), int(center_x + area), int(center_y - area), int(center_y + area)
    print(x1, x2, y1, y2,img.shape[0],img.shape[1])

    if x1 < 0 : x1 = 0
    if x1 > img.shape[0]:x1 = img.shape[0]-(x1 - img.shape[0])
    if x2 < 0 : x2 = 0
    if x2 > img.shape[0]:x2 = img.shape[0] - (x2 - img.shape[0])
    if y1 < 0 : y1 = 0
    if y1 > img.shape[1]:y1 = img.shape[1]
    if y2 < 0 : y2 = 0
    if y2 > img.shape[1]:y2 = img.shape[1]
   
    print(x1, x2, y1, y2,img.shape[0],img.shape[1])
    bat_crop = img[x1: x2, y1: y2]
    return bat_crop
    
def find_bats(allImgs):
    totalBats = 0
    batDepthMin = 50
    batDepthMax = 400
    cropped_bats = []
    bats = []

    for img in allImgs:
        blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]  # Find bats using cv.findContours

        # Process the blobs
        for blob in blobs:
            blobNum = 0
            if batDepthMin < cv.contourArea(blob) < batDepthMax:  # Only process blobs with a min / max size
                rect = cv.minAreaRect(blob)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                M = cv.moments(blob)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bat_cords = (cx, cy)
                bats.append(bat_cords)
                # crop bat from image
                cv.drawContours(img[0], [box], 0, (0, 0, 255), 1)  # Drawing on image before cropping to see bats
                cropped_bat = crop_bat(img[0], cx, cy)
                if blobNum < 10:
                    cv.imshow("cropped bat {}".format(blobNum), cropped_bat)
                    cv.waitKey(0)
                cropped_bats.append((cropped_bat, bat_cords))
                # cv.drawContours(img[0], [box], 0, (0, 0, 255), 1)  # Draw bat contours on img
            blobNum += 1

    totalBats += len(bats)
        
    return allImgs, totalBats, cropped_bats

