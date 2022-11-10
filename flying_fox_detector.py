import cv2 as cv
from functions import  chop_img, find_bats
from fastai.vision.all import *
import pathlib
import torch
import torch.nn 
import time

start = time.time()
img = cv.imread(r"DJI_0017_T.JPG")
totalBats = 0

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

cropped_bats = []
for img in allImgs:
    cropped_bats.extend(find_bats(img))

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learn = torch.load('model.pkl')
# learn = torch.nn.DataParallel(learn)
# learn.to(device)

learn = load_learner("model_densenet_zeros.pkl", cpu=True) # Change to cpu=False if using a GPU
if __name__ == '__main__':
    test_dl = learn.dls.test_dl(cropped_bats[:50])
    preds = learn.get_preds(dl=test_dl)

    # print(preds)
    # results = [tuple(f"{pred[0]:.4f}", f"{pred[1]:.4f}") for pred in preds]

    results = [tuple([f"{pred[0]:.4f}",f"{pred[1]:.4f}"]) for pred in preds[0]]
    for result in results:
        print(result)

# print(f"Time before Detecting bats took {time.time() - start:.2f} seconds")
# print("Detecting Bats...")

# start1 = time.time()


# def predict(bat):
#     with learn.no_bar(), learn.no_logging():
#         _, _, probs = learn.predict(bat)
#     return (tuple(map(lambda x: f"{x:.4f}", probs)))



# results = [predict(bat) for bat in cropped_bats]

# filtered_results = filter(lambda x: (x[0] < '0.5' and x[1] > '0.75'), results)

# print(len(list(filtered_results)))
# print(f"Detecting Bats took: {time.time()-start1:.2f} seconds")

