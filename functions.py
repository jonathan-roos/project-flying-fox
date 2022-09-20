import cv2 as cv
import numpy as np
import keyboard


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
    
def crop_bat(img, box, index):
    x1, y1 = int(box[0][1]), int(box[0][0])
    x2, y2 = int(box[1][1]), int(box[1][0]) 
    x3, y3 = int(box[2][1]), int(box[2][0])
    x4, y4 = int(box[3][1]), int(box[3][0])

    # Find distance from cx to top_left_x and cy to top_left_y to determine how many pixels the border around the cropped image should be
    top_left_x = min([x1,x2,x3,x4])
    top_left_y = min([y1,y2,y3,y4])
    bot_right_x = max([x1,x2,x3,x4])
    bot_right_y = max([y1,y2,y3,y4])
    
    bat_crop = img[top_left_x-10: bot_right_x+11, top_left_y-10: bot_right_y+11]

    label_bat(bat_crop, img, box, index)

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
        i =0
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
                
                cropped_bat = crop_bat(img[0], box, i)
                cropped_bats.append((cropped_bat, bat_cords, box))
                i+=1

    totalBats += len(bats)
        
    return allImgs, totalBats, cropped_bats

def label_bat(bat_crop, img, box, i):
    num = i + 145
    cv.imshow("cropped_bat", bat_crop)
    img_dup = img.copy()
    cv.drawContours(img_dup, [box], 0, (0, 0, 255), 1)  # Draw bat contours on img original
    cv.imshow("context", img_dup)
    
    cv.waitKey(0)
    
    if keyboard.is_pressed('y'):
        print("y was pressed")
        path = r"C:\Users\jonathan\OneDrive - Evolve Technology\Documents\Project Flying Fox\Training Data\bat\bat{}.png".format(num)
        cv.imwrite(path, bat_crop)

    if keyboard.is_pressed('n'):
        print("n was pressed")
        path = r"C:\Users\jonathan\OneDrive - Evolve Technology\Documents\Project Flying Fox\Training Data\!bat\bat{}.png".format(num)
        cv.imwrite(path, bat_crop)
    
    
    cv.destroyAllWindows()
