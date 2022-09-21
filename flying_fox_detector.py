import cv2 as cv
import numpy as np
from functions import  chop_img, find_bats
from fastai.vision.all import *

import pathlib


img_nums = ["0902", "0690", "0217"]
img_num = img_nums[2]
img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_{}_T.JPG".format(img_num))
print(img.shape)  

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

# For each image, use cv.findcontours on threshed img and cv.drawcontours on original image
allImgs, totalBats, cropped_bats = find_bats(allImgs)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("model.pkl")
bat_count = 0
# not_bat_count = 0


for bat in cropped_bats:
    img = bat[0]
    label, _, probs = learn.predict(img)
    p=f"{probs[1]:.4f}"
    if label == 'bat' and p>'0.5':
        bat_count += 1
    # elif label == "!bat":
    #     not_bat_count += 1

print(f"Number of bats: {bat_count}")
# print(f"Number of !bats = {not_bat_count}")

               