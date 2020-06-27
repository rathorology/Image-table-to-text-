import cv2
import pytesseract
import numpy as np
import pandas as pd
import random
import kraken as k
import warnings
from kraken.kraken.lib.models import load_any
from kraken.kraken import rpred
from PIL import Image
from subprocess import call

warnings.filterwarnings("ignore", category=FutureWarning)

##Kraken Model Loading
model = load_any("en-default.mlmodel")

img = cv2.imread('images/table_image.png')
genrator_image = Image.fromarray(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- performing Otsu threshold ---
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# cv2.imshow('thresh1', thresh1)
# cv2.waitKey(0)


rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
# cv2.imshow('dilation', dilation)
# cv2.waitKey(0)


# ---Finding contours ---
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
box_list = []

row_list = list()
i = 0
one_row = list()
for idx, cnt in enumerate(contours[::-1]):
    x, y, w, h = cv2.boundingRect(cnt)
    box_list.append([x, y, w, h])
    if idx == 0:
        one_row.append([x, y, w, h])
    else:
        if y - i < 15:
            one_row.append([x, y, w, h])
        else:
            row_list.append(one_row)
            one_row = list()
            one_row.append([x, y, w, h])
    i = y
# print(row_list)
color = (np.random.random(size=3) * 256)
im2 = img.copy()
all_text = list()
count_in_each_row = list()
# bound = {'boxes': box_list, 'text_direction': 'horizontal-lr'}
# ##Use this for using Kraken API , uncomment model from above
# generator = next(rpred.rpred(network=model, im=img1, bounds=bound))
# box_text = generator.prediction
for row_boxes in row_list:

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    row_text = list()
    for one_box in row_boxes:
        x, y, w, h = one_box
        cv2.rectangle(im2, (x, y), (x + w, y + h), (b, g, r), 3)

        small = img[y:y + h, x:x + w]
        # box_text = pytesseract.image_to_string(small, lang='eng')
        bound = {'boxes': [tuple(one_box)], 'text_direction': 'vertical-lr'}
        ##Use this for using Kraken API , uncomment model from above
        generator = rpred.rpred(network=model, im=genrator_image, bounds=bound)
        nxt_gen = next(generator)
        box_text = nxt_gen.prediction
        print(nxt_gen.prediction, nxt_gen.confidences, nxt_gen.cuts)

        # cv2.imwrite("images/temp.jpg", small)
        # box_text = " "
        # try:
        #     call(["kraken", "-i", "images/temp.jpg", "image.txt", "binarize", "segment", "ocr"])
        #     box_text = open("image.txt", "r").read()
        # except Exception as e:
        #     pass

        if box_text == '':
            box_text = " "
        row_text.append(box_text)
        print("Box_Text = {} | Y = {}".format(box_text, y))
    print(row_text)
    count_in_each_row.append(len(row_text))
    all_text.append(row_text)
    print("======================================================================")
print(all_text)
updated_text_rows = list()
# Creating a dataframe of the generated OCR list
for rows in all_text:
    diff = max(count_in_each_row) - len(rows)
    rows = rows + [" "] * diff
    updated_text_rows.append(rows)

arr = np.array(updated_text_rows)
dataframe = pd.DataFrame(arr, columns=range(0, max(count_in_each_row)))
dataframe.to_csv("output.csv", index=False)

cv2.imshow('final', im2)
cv2.waitKey(0)
# print(len(box))
# print(len(text_list))
