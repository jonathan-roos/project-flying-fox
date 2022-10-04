import cv2 as cv
from functions import  chop_img, find_bats
from fastai.vision.all import *
import pathlib
import time

start = time.time()
img_nums = ("0800", "0690", "0217")
img_num = img_nums[2]
img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_{}_T.JPG".format(img_num))
totalBats = 0

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

# For each image, use cv.findcontours on threshed img and cv.drawcontours on original image
# cropped_bats = find_bats(allImgs)
cropped_bats = []
for img in allImgs:
    cropped_bats.extend(find_bats(img))


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("model.pkl")
print(f"Time before Detecting bats took {time.time() - start:.2f} seconds")
print("Detecting Bats...")

start1 = time.time()

# def predict(bat):
#     labels = ("!bat", "bat")
#     with learn.no_bar(), learn.no_logging():
#         _, _, probs = learn.predict(bat)
#     return (tuple(zip(labels, map(lambda x: f"{x:.4f}", probs))))
def predict(bat):
    with learn.no_bar(), learn.no_logging():
        _, _, probs = learn.predict(bat)
    return (tuple(map(lambda x: f"{x:.4f}", probs)))

results = [predict(bat) for bat in cropped_bats]

# def filter_bats(results):
#     if results[0][1] < '0.5' and results[1][1] > '0.75':
#         return results

filtered_results = filter(lambda x: (x[0] < '0.5' and x[1] > '0.75'), results)

# for result in results:
#     if result[0][1] < '0.5' and result[1][1] > '0.75':
#         totalBats += 1

print(len(list(filtered_results)))
print(f"Detecting Bats took: {time.time()-start1:.2f} seconds")


