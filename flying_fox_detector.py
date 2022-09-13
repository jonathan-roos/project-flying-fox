import cv2 as cv
import numpy as np
import keyboard
from functions import augment, chop_img, find_bats

img_nums = ["0265", "0794", "0217"]
img_num = img_nums[1]
img = cv.imread(r"C:\Users\jonathan\Evolve Technology\Evolve Technologies Team Site - Client Info\Ecosure\4. Projects\Project Flying Fox - Sample Data\PR5902 Hillview Station Apr 2022\Raw Data M2EA 270422\Ortho Runs\40M\Thermal\DJI_{}_T.JPG".format(img_num))
print(img.shape)   
# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

# For each image, use cv.findcontours on threshed img and cv.drawcontours on original image
allImgs, totalBats, cropped_bats = find_bats(allImgs)

# Save cropped bats to file
for i in range(20):
    bat = cropped_bats[100+i][0]
    cv.imshow("cropped bat {}".format(i), bat)
    path = r"C:\Users\jonathan\OneDrive - Evolve Technology\Documents\Project Flying Fox\croppedBats\bat\bat{}.png".format(i)

    if keyboard.is_pressed('y'):
        path = path
    if keyboard.is_pressed('n'):
        path = r"C:\Users\jonathan\OneDrive - Evolve Technology\Documents\Project Flying Fox\croppedBats\!bat\bat{}.png".format(i)

    cv.waitKey(0)
    cv.imwrite(path, bat)
    
# Concatonate the marked images back together
img_row_1 = cv.hconcat([allImgs[0][0],allImgs[1][0],allImgs[2][0]])
img_row_2 = cv.hconcat([allImgs[3][0],allImgs[4][0],allImgs[5][0]])
img_row_3 = cv.hconcat([allImgs[6][0],allImgs[7][0],allImgs[8][0]])
img_concat = cv.resize(cv.vconcat([img_row_1, img_row_2, img_row_3]), (960, 768))

text = "Bats detected: {}".format(totalBats)
cv.putText(img_concat, text, (350, 750), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

print("Total number of bats = {}".format(totalBats))
cv.imshow("img", img_concat)
cv.waitKey(0)

               