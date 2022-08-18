import cv2 as cv
import numpy as np


img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_0203_T.JPG")
kernel = np.ones((5, 5), np.uint8)

print(img.shape)
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
        # allImgs.append((imgResize,imgThresh))
        allImgs.append((imgResize,imgDilationInv))
        j += 1
    i += 1

# Find bats using cv.SimpleBlobDetector
blobParams = cv.SimpleBlobDetector_Params()

blobParams.filterByArea = False
blobParams.minArea = 15
blobParams.maxArea = 100

blobParams.filterByCircularity = False
blobParams.minCircularity = 0.85

blobParams.filterByConvexity = False
blobParams.maxConvexity = 1

blobParams.filterByInertia = False
blobParams.maxInertiaRatio = 1

detector = cv.SimpleBlobDetector_create(blobParams)
for img in allImgs:
    imageUsed = img[1]

    keypoints = detector.detect(imageUsed)
    print(keypoints[0].pt)

    blank = np.zeros((1, 1))
    batsThreshed = cv.drawKeypoints(img[1], keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    batsOriginal = cv.drawKeypoints(img[0], keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    numOfBats = len(keypoints)
    text = "Bats detected: {}".format(str(len(keypoints)))
    cv.putText(batsThreshed, text, (200, 500), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.putText(batsOriginal, text, (200, 500), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # cv.imshow("Threshed 1", img[1])
    # cv.imshow("Original", img[0])
    cv.imshow("Bats detected", batsOriginal)
    cv.imshow("Bats threshed", batsThreshed)
    cv.waitKey(0)

