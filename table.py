import numpy as np
import cv2 as cv
from util import verify_table, isolate_lines, Table

image = cv.imread("images/electricity_2.png")

# Convert resized RGB image to grayscale
NUM_CHANNELS = 3
if len(image.shape) == NUM_CHANNELS:
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# =====================================================
# IMAGE FILTERING (using adaptive thresholding)
# =====================================================
"""
ADAPTIVE THRESHOLDING
Thresholding changes pixels' color values to a specified pixel value if the current pixel value
is less than a threshold value, which could be:

1. a specified global threshold value provided as an argument to the threshold function (simple thresholding),
2. the mean value of the pixels in the neighboring area (adaptive thresholding - mean method),
3. the weighted sum of neigborhood values where the weights are Gaussian windows (adaptive thresholding - Gaussian method).

The last two parameters to the adaptiveThreshold function are the size of the neighboring area and
the constant C which is subtracted from the mean or weighted mean calculated.
"""
MAX_THRESHOLD_VALUE = 255
BLOCK_SIZE = 15
THRESHOLD_CONSTANT = 0

# Filter image
filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                BLOCK_SIZE, THRESHOLD_CONSTANT)

# =====================================================
# LINE ISOLATION
# =====================================================
"""
HORIZONTAL AND VERTICAL LINE ISOLATION
To isolate the vertical and horizontal lines, 

1. Set a scale.
2. Create a structuring element.
3. Isolate the lines by eroding and then dilating the image.
"""
SCALE = 15

# Isolate horizontal and vertical lines using morphological operations
horizontal = filtered.copy()
vertical = filtered.copy()

horizontal_size = int(horizontal.shape[1] / SCALE)
horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
isolate_lines(horizontal, horizontal_structure)

vertical_size = int(vertical.shape[0] / SCALE)
vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
isolate_lines(vertical, vertical_structure)

# =====================================================
# TABLE EXTRACTION
# =====================================================
# Create an image mask with just the horizontal
# and vertical lines in the image. Then find
# all contours in the mask.
mask = horizontal + vertical
(contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find intersections between the lines
# to determine if the intersections are table joints.
intersections = cv.bitwise_and(horizontal, vertical)

# Get tables from the images
ROI_number = 0
tables = []  # list of tables
for i in range(len(contours)):
    # Verify that region of interest is a table
    (rect, table_joints) = verify_table(contours[i], intersections)
    if rect == None or table_joints == None:
        continue

    # Create a new instance of a table
    table = Table(rect[0], rect[1], rect[2], rect[3])

    # Get an n-dimensional array of the coordinates of the table joints
    joint_coords = []
    for i in range(len(table_joints)):
        joint_coords.append(table_joints[i][0][0])
    joint_coords = np.asarray(joint_coords)

    # Returns indices of coordinates in sorted order
    # Sorts based on parameters (aka keys) starting from the last parameter, then second-to-last, etc
    sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
    joint_coords = joint_coords[sorted_indices]

    # Store joint coordinates in the table instance
    table.set_joints(joint_coords)

    tables.append(table)
    x = table.x
    x1 = table.x + table.w
    y = table.y
    y1 = table.y + table.h
    cv.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)

    ROI = image[y:y1, x:x1]
    cv.imwrite('image_tables/ROI_{}.png'.format(ROI_number), ROI)
    ROI_number += 1
    cv.imshow("tables", ROI)
    cv.waitKey(0)
