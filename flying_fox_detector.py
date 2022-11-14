import cv2 as cv
from functions import  chop_img, find_bats
from fastai.vision.all import *
import pathlib
import torch
import torch.nn 
import time

start = time.time()
img = cv.imread(r"DJI_0017_T.JPG")

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

cropped_bats = []
for img in allImgs:
    cropped_bats.extend(find_bats(img))

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

start = time.time()
learn = load_learner("model_densenet_zeros.pkl", cpu=True) # Change to cpu=False if using a GPU

# getting batch predictions using learn.get_preds() instead of using learn.predict().
# results in much quicker prediction times
if __name__ == '__main__':
    test_dl = learn.dls.test_dl(cropped_bats)
    preds = learn.get_preds(dl=test_dl)

    results = [tuple([f"{pred[0]:.4f}",f"{pred[1]:.4f}"]) for pred in preds[0]]

    filtered_results = filter(lambda x: (x[0] < '0.5' and x[1] > '0.75'), results)

    print(len(list(filtered_results)))
    print(f"Detecting Bats took: {time.time()-start:.2f} seconds")

