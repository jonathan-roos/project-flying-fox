import cv2 as cv
import numpy as np
from functions import  chop_img, find_bats
from fastai.vision.all import *

import pathlib


img_nums = ("0115", "0690", "0217")
img_num = img_nums[2]
img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_{}_T.JPG".format(img_num))

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("model.pkl")

# For each image, use cv.findcontours on threshed img and cv.drawcontours on original image
allImgsBefore, allImgsAfter, totalBats = find_bats(allImgs, learn)

img_row_before_1 = cv.hconcat([allImgsBefore[0],allImgsBefore[1],allImgsBefore[2]])
img_row_before_2 = cv.hconcat([allImgsBefore[3],allImgsBefore[4],allImgsBefore[5]])
img_row_before_3 = cv.hconcat([allImgsBefore[6],allImgsBefore[7],allImgsBefore[8]])
img_concat_before = cv.resize(cv.vconcat([img_row_before_1, img_row_before_2, img_row_before_3]), (960, 768))

img_row_after_1 = cv.hconcat([allImgsAfter[0],allImgsAfter[1],allImgsAfter[2]])
img_row_after_2 = cv.hconcat([allImgsAfter[3],allImgsAfter[4],allImgsAfter[5]])
img_row_after_3 = cv.hconcat([allImgsAfter[6],allImgsAfter[7],allImgsAfter[8]])
img_concat_after = cv.resize(cv.vconcat([img_row_after_1, img_row_after_2, img_row_after_3]), (960, 768))
cv.imshow("Before NN", img_concat_before)
cv.imshow("After NN", img_concat_after)
cv.waitKey(0)

print(f"Number of bats: {totalBats}")

               