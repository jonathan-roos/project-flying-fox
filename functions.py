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

# def crop_bat(img, box):
#     y1 = int((box[0][0]) * 0.975)
#     x1 = int((box[0][1]) * 0.975)
#     y2 = int((box[2][0]) * 1.025)
#     x2 = int((box[2][1]) * 1.025)
#     bat_crop = img[x1: x2, y1: y2]

#     print(box)
#     print(x1, y1, x2, y2)
#     print(bat_crop.shape)
#     # bat_crop = cv.resize(bat_crop, (bat_crop.shape[0]*10, bat_crop.shape[1]*10))
#     return bat_crop
    
def crop_bat(img, box):
    x1, y1 = int(box[0][1]), int(box[0][0])
    x2, y2 = int(box[1][1]), int(box[1][0]) 
    x3, y3 = int(box[2][1]), int(box[2][0])
    x4, y4 = int(box[3][1]), int(box[3][0])

    top_left_x = min([x1,x2,x3,x4])
    top_left_y = min([y1,y2,y3,y4])
    bot_right_x = max([x1,x2,x3,x4])
    bot_right_y = max([y1,y2,y3,y4])

    bat_crop = img[top_left_x: bot_right_x+1, top_left_y: bot_right_y+1]

    print(box)
    print(x1, y1, x2, y2)
    print(bat_crop.shape)
    # bat_crop = cv.resize(bat_crop, (bat_crop.shape[0]*10, bat_crop.shape[1]*10))
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
                cropped_bat = crop_bat(img[0], box)
                cv.drawContours(img[0], [box], 0, (0, 0, 255), 1)  # Draw bat contours on img original
                cv.drawContours(img[1], [box], 0, (0, 0, 255), 1)  # Draw bat contours on img aug

                # cv.imshow("img original", img[0])
                # cv.imshow("img aug", img[1])

                # cv.imshow("cropped bat", cropped_bat)
                # cv.waitKey(0)
                cropped_bats.append((cropped_bat, bat_cords))
                # cv.drawContours(img[0], [box], 0, (0, 0, 255), 1)  # Draw bat contours on img

    totalBats += len(bats)
        
    return allImgs, totalBats, cropped_bats

