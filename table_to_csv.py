import cv2
import pytesseract
import numpy as np
import pandas as pd
import random
import kraken as k
import warnings
from kraken.lib.models import load_any
from kraken import rpred, binarization
from PIL import Image
from subprocess import call

warnings.filterwarnings("ignore", category=FutureWarning)

## ---Loading Kraken Model---
model = load_any("en-default.mlmodel")

img = cv2.imread('images/tabualr.png')

## ---Binarization of image---
genrator_image = Image.fromarray(img)
genrator_image = binarization.nlbin(genrator_image)

# ----Grayscaling Image----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- performing Otsu threshold ---
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
cv2.imwrite("processed_image/threshold.png", thresh1)
# cv2.imshow('thresh1', thresh1)
# cv2.waitKey(0)

# ----Image dialation----
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
cv2.imwrite("processed_image/dilation.png", dilation)
# cv2.imshow('dilation', dilation)
# cv2.waitKey(0)


# ---Finding contours ---
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dummy_image = img.copy()
row_list = list()
old_y = 0
single_row = list()
for idx, cnt in enumerate(contours[::-1]):
    area = cv2.contourArea(cnt)

    if area < 25000:
        x, y, w, h = cv2.boundingRect(cnt)

        ## Different color for each row
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Drawing box
        cv2.rectangle(dummy_image, (x, y), (x + w, y + h), (b, g, r), 2)
        if idx == 0:
            single_row.append([x, y, w, h])
        else:
            if y - old_y < 5:
                single_row.append([x, y, w, h])
            else:
                row_list.append(single_row)
                single_row = list()
                single_row.append([x, y, w, h])
        old_y = y

cv2.imwrite("processed_image/show_box.png", dummy_image)
# cv2.imshow('final', dummy_image)
# cv2.waitKey(0)


# color = (np.random.random(size=3) * 256)

all_text = list()
row_count = list()

for row_boxes in row_list:
    row_text = list()
    for one_box in row_boxes:
        x, y, w, h = one_box
        #########################################################################################

        ## Kraken Text Extraction
        cord = [x, y, x + w, y + h]
        bound = {'boxes': [tuple(cord)], 'text_direction': 'horizontal-lr'}
        ##Use this for using Kraken API , uncomment model from above
        generator = rpred.rpred(network=model, im=genrator_image, bounds=bound)
        nxt_gen = next(generator)
        box_text = nxt_gen.prediction

        ##########################################################################################

        ## Kraken bash script
        # small = img[y:y + h, x:x + w]
        # cv2.imwrite("images/temp.jpg", small)
        # box_text = " "
        # try:
        #     call(["kraken", "-i", "images/temp.jpg", "image.txt", "binarize", "segment", "ocr"])
        #     box_text = open("image.txt", "r").read()
        # except Exception as e:
        #     pass

        row_text.append(box_text)
        print("Box_Text = {} | Y = {}".format(box_text, y))
    print(row_text)
    row_count.append(len(row_text))
    all_text.append(row_text)
    print("======================================================================")
print(all_text)
updated_text_rows = list()

columns = max(set(row_count), key=row_count.count)

for rows in all_text:
    diff = columns - len(rows)
    rows = rows + [" "] * diff
    updated_text_rows.append(rows)

# Creating a dataframe of the generated OCR list
arr = np.array(updated_text_rows)
dataframe = pd.DataFrame(arr, columns=range(0, columns))
dataframe.to_csv("output_csv/output.csv", index=False)
