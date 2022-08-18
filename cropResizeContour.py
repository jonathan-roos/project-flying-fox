import cv2 as cv
import numpy as np


img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_0203_T.JPG")
kernel = np.ones((5, 5), np.uint8)

height = 512
width = 640
ratio = 0.8
step = 3

i = 0
j = 0


allImgs = []
for i in range(step):
    for j in range(step):
        imgCrop = img[int(0 + height/step * i) : int(height/step + height/step * i),
                int(0 + width/step * j) : int(width/step + width/step * j)]
        imgResize = cv.resize(imgCrop, (640, 512))
        imgGray = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)
        imgBlur = cv.GaussianBlur(imgGray, (7, 7), 0)
        imgThresh = cv.adaptiveThreshold(imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 5)
        imgDilation = cv.dilate(imgThresh, kernel, iterations=1)
        imgDilationInv = cv.bitwise_not(imgDilation)
        allImgs.append((imgResize,imgDilation))
        j += 1
    i += 1

# Find bats using cv.findContours
totalBats = 0
for img in allImgs:
    blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]

    # Bat depth!
    batDepthMin = 35
    batDepthMax = 400
    bats = []

    # Process the blobs
    for blob in blobs:
        if batDepthMin < cv.contourArea(blob) < batDepthMax:  # Only process blobs with a min / max size
            bats.append(blob)

            # Circle drawing maths shit
            M = cv.moments(blob)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # cv.circle(original, (cx, cy), 6, (255, 0, 0), 1)

    print("Number of bats = {}".format(len(bats)))
    totalBats += len(bats)
    # Does what it says it does
    cv.drawContours(img[0], bats, -1, (0, 255, 0), 1)


    cv.imshow("original", img[0])
    cv.imshow("cropped", img[1])
    cv.waitKey(0)
print("Total number of bats = {}".format(totalBats))
