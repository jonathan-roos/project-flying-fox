from timeit import default_timer as timer
import cv2 as cv
# from functions import  chop_img, find_bats
from fastai.vision.all import *
import pathlib 

def chop_img(img, step):
    allImgs = []
    width, height, i, j = 640, 512, 0, 0
    for i in range(step):
        for j in range(step):
            # crop the image into step*rows and step*columns
            imgCrop = img[int(0 + height / step * i): int(height / step + height / step * i),
                      int(0 + width / step * j): int(width / step + width / step * j)]
            imgGray = cv.cvtColor(imgCrop, cv.COLOR_BGR2GRAY)
            imgResize = cv.resize(imgGray, (640, 512))  # Resize image
            imgAug = augment(imgResize)
            allImgs.append((imgResize, imgAug))
            j += 1
        i += 1
    return allImgs

def augment(img):
    kernel = np.ones((5, 5), np.uint8)
    # imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
    imgBlur = cv.GaussianBlur(img, (5, 5), 0)  # apply gaussian blur
    imgThresh = cv.adaptiveThreshold(  # apply adaptive threshold
    imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 5)
    imgDilation = cv.dilate(imgThresh, kernel, iterations=1)  # apply dilation to amplify threshold result
    return imgDilation
    
def crop_bat(img, box):
    x1, y1, x2, y2, x3, y3, x4, y4 = int(box[0][1]), int(box[0][0]), int(box[1][1]), int(box[1][0]), int(box[2][1]), int(box[2][0]), int(box[3][1]), int(box[3][0])

    # Find distance from cx to top_left_x and cy to top_left_y to determine how many pixels the border around the cropped image should be
    top_left_x, top_left_y, bot_right_x, bot_right_y = min([x1,x2,x3,x4]), min([y1,y2,y3,y4]), max([x1,x2,x3,x4]), max([y1,y2,y3,y4])

    crop_x1 = top_left_x - 10 
    if crop_x1 <= 0:
        crop_x1 = 1 
    
    crop_x2 = bot_right_x+11
    if crop_x2 > 512:
        crop_x2 = 512 

    crop_y1 = top_left_y-10
    if crop_y1 <= 0:
        crop_y1 = 1

    crop_y2 = bot_right_y+11
    if crop_y2 > 640:
        crop_y2 = 640 

    bat_crop = img[crop_x1: crop_x2, crop_y1: crop_y2]
    return bat_crop

    
def find_bats(allImgs, learn):
    totalBats = 0
    batDepthMin = 50
    batDepthMax = 400
    allImgBefore = []
    allImgAfter = []

    for img in allImgs:
        blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]
        before = img[0].copy()
        after = img[0].copy()
        # Process the blobs
        for blob in blobs:
            if batDepthMin < cv.contourArea(blob) < batDepthMax:  # Only process blobs with a min / max size 
                rect = cv.minAreaRect(blob)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(before, [box], 0, (0,0,255),1)
                cropped_bat = crop_bat(img[0], box)
                with learn.no_bar(), learn.no_logging():
                    label, _, probs = learn.predict(cropped_bat)
                p_not_bat=f"{probs[0]:.4f}"
                p_bat=f"{probs[1]:.4f}"
                if p_not_bat < '0.9' and p_bat > '0.5':
                    cv.drawContours(after, [box], 0, (0,0,255),1)
                    totalBats += 1
        allImgBefore.append(before)
        allImgAfter.append(after)
                
    return allImgBefore, allImgAfter, totalBats

# img_nums = ("0045", "0690", "0217")
# img_num = img_nums[0]
# img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_{}_T.JPG".format(img_num))

user_input = input("Please enter thermal image path: ")
img = cv.imread(user_input)

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("model_densenet_zeros.pkl")
# For each image, use cv.findcontours on threshed img and cv.drawcontours on original image
start = timer()
allImgsBefore, allImgsAfter, totalBats = find_bats(allImgs, learn)
end = timer()
img_row_before_1 = cv.hconcat([allImgsBefore[0],allImgsBefore[1],allImgsBefore[2]])
img_row_before_2 = cv.hconcat([allImgsBefore[3],allImgsBefore[4],allImgsBefore[5]])
img_row_before_3 = cv.hconcat([allImgsBefore[6],allImgsBefore[7],allImgsBefore[8]])
img_concat_before = cv.resize(cv.vconcat([img_row_before_1, img_row_before_2, img_row_before_3]), (960, 768))

img_row_after_1 = cv.hconcat([allImgsAfter[0],allImgsAfter[1],allImgsAfter[2]])
img_row_after_2 = cv.hconcat([allImgsAfter[3],allImgsAfter[4],allImgsAfter[5]])
img_row_after_3 = cv.hconcat([allImgsAfter[6],allImgsAfter[7],allImgsAfter[8]])
img_concat_after = cv.resize(cv.vconcat([img_row_after_1, img_row_after_2, img_row_after_3]), (960, 768))
end = timer()
print(end-start)
cv.imshow("Before NN", img_concat_before)
cv.imshow("After NN", img_concat_after)
cv.waitKey(0)

print(f"Number of bats: {totalBats} found in {end-start:.4f}s")

