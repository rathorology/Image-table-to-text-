import cv2
import numpy as np
import pandas as pd
import random
from kraken.lib.models import load_any
from kraken import rpred, binarization
from PIL import Image
from subprocess import call
from imutils import contours
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

## ---Loading Kraken Model---
model = load_any("en-default.mlmodel")


def preprocessing_non_tabular(path):
    img = cv2.imread(path)

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
    return img, genrator_image, contours[::-1]


def preprocessing_tabular(path):
    # Load image
    img = cv2.imread(path)

    ## ---Binarization of image---
    genrator_image = Image.fromarray(img)
    genrator_image = binarization.nlbin(genrator_image)

    # ----Grayscaling Image----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- performing Otsu threshold ---
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove text characters with morph open and contour filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:
            cv2.drawContours(opening, [c], -1, (0, 0, 0), -1)

    # Repair table lines, sort contours, and extract ROI
    close = 255 - cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="top-to-bottom")
    return img, genrator_image, cnts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR on Tablular Image")
    parser.add_argument('--img-path', type=str, help='path to your image.')
    args = parser.parse_args()

    # ---Image_Path---
    path = args.img_path
    # path = "images/patient.png"

    img, genrator_image, cnts = preprocessing_non_tabular(path)
    if len(cnts) < 8:
        img, genrator_image, cnts = preprocessing_tabular(path)

    row_list = list()
    old_y = 0
    single_row = list()
    for idx, cnt in enumerate(cnts):

        x, y, w, h = cv2.boundingRect(cnt)
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

    # color = (np.random.random(size=3) * 256)
    dummy_image = img.copy()

    all_text = list()
    row_count = list()

    for row_boxes in row_list:
        row_text = list()
        for one_box in row_boxes:
            x, y, w, h = one_box
            ## Different color for each row
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            # Drawing box
            cv2.rectangle(dummy_image, (x, y), (x + w, y + h), (b, g, r), 2)
            #########################################################################################

            ## Kraken Text Extraction
            cord = [x, y, x + w, y + h]
            bound = {'boxes': [tuple(cord)], 'text_direction': 'horizontal-lr'}

            ## Using Kraken API
            generator = rpred.rpred(network=model, im=genrator_image, bounds=bound)
            nxt_gen = next(generator)
            box_text = nxt_gen.prediction
            print("Box_Text = {} | Y = {}".format(box_text, y))

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

        print(row_text)
        row_count.append(len(row_text))
        all_text.append(row_text)
        print("======================================================================")
    print(all_text)

    cv2.imwrite("processed_image/show_box.png", dummy_image)
    # cv2.imshow('final', dummy_image)
    # cv2.waitKey(0)

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
