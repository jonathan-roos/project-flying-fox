from timeit import default_timer as timer
import cv2 as cv
from functions import  chop_img, find_bats
from fastai.vision.all import *
import pathlib
import timm 

start = timer()
img_nums = ("0800", "0690", "0217")
img_num = img_nums[2]
img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_{}_T.JPG".format(img_num))
totalBats = 0

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

# For each image, use cv.findcontours on threshed img and cv.drawcontours on original image
cropped_bats = find_bats(allImgs)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("model_convnext_small.pkl")

i = 0
while i < len(cropped_bats):
    label, _, probs = learn.predict(cropped_bats[i])
    p=f"{probs[0]:.4f}"
    if label == '!bat' and p > '0.5':
        pass
    else:
        totalBats += 1
    i += 1


print(timer()-start)
print(f"Number of bats: {totalBats}")
