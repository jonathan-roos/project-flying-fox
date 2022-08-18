import cv2 as cv
import numpy as np
from functions import augment, chop_img, find_bats

img_num = "0137"
img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_{}_T.JPG".format(img_num))
print(img.shape)
# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

# For each image, use cv.findcontours on threshed img and cv.drawcontours on original image
allImgs, totalBats, bat_cords = find_bats(allImgs)

# Concatonate the marked images back together
img_row_1 = cv.hconcat([allImgs[0][0],allImgs[1][0],allImgs[2][0]])
img_row_2 = cv.hconcat([allImgs[3][0],allImgs[4][0],allImgs[5][0]])
img_row_3 = cv.hconcat([allImgs[6][0],allImgs[7][0],allImgs[8][0]])


# Resize img
img_concat = cv.resize(cv.vconcat([img_row_1, img_row_2, img_row_3]), (960, 768))

text = "Bats detected: {}".format(totalBats)
cv.putText(img_concat, text, (350, 750), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

print("Total number of bats = {}".format(totalBats))
cv.imshow("img", img_concat)
cv.waitKey(0)
