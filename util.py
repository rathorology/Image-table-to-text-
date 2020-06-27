import cv2 as cv
import numpy as np


class Table:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.joints = None

    def __str__(self):
        return "(x: %d, y: %d, w: %d, h: %d)" % (self.x, self.x + self.w, self.y, self.y + self.h)

    # Stores the coordinates of the table joints.
    # Assumes the n-dimensional array joints is sorted in ascending order.
    def set_joints(self, joints):
        if self.joints != None:
            raise ValueError("Invalid setting of table joints array.")

        self.joints = []
        row_y = joints[0][1]
        row = []
        for i in range(len(joints)):
            if i == len(joints) - 1:
                row.append(joints[i])
                self.joints.append(row)
                break

            row.append(joints[i])

            # If the next joint has a new y-coordinate,
            # start a new row.
            if joints[i + 1][1] != row_y:
                self.joints.append(row)
                row_y = joints[i + 1][1]
                row = []

    # Prints the coordinates of the joints.
    def print_joints(self):
        if self.joints == None:
            print("Joint coordinates not found.")
            return

        print("[")
        for row in self.joints:
            print("\t" + str(row))
        print("]")

    # Finds the bounds of table entries in the image by
    # using the coordinates of the table joints.
    def get_table_entries(self):
        if self.joints == None:
            print("Joint coordinates not found.")
            return

        entry_coords = []
        for i in range(0, len(self.joints) - 1):
            entry_coords.append(self.get_entry_bounds_in_row(self.joints[i], self.joints[i + 1]))

        return entry_coords

    # Finds the bounds of table entries
    # in each row based on the given sets of joints.
    def get_entry_bounds_in_row(self, joints_A, joints_B):
        row_entries = []

        # Since the sets of joints may not have the same
        # number of points, we pick the set with a lower number
        # of points to find the bounds from.
        if len(joints_A) <= len(joints_B):
            defining_bounds = joints_A
            helper_bounds = joints_B
        else:
            defining_bounds = joints_B
            helper_bounds = joints_A

        for i in range(0, len(defining_bounds) - 1):
            x = defining_bounds[i][0]
            y = defining_bounds[i][1]
            w = defining_bounds[i + 1][0] - x  # helper_bounds's (i + 1)th coordinate may not be the lower-right corner
            h = helper_bounds[0][1] - y  # helper_bounds has the same y-coordinate for all of its elements

            # If the calculated height is less than 0,
            # make the height positive and
            # use the y-coordinate of the row above for the bounds
            if h < 0:
                h = -h
                y = y - h

            row_entries.append([x, y, w, h])

        return row_entries


"""
Apply morphology operations
"""


def isolate_lines(src, structuring_element):
    cv.erode(src, structuring_element, src, (-1, -1))  # makes white spots smaller
    cv.dilate(src, structuring_element, src, (-1, -1))  # makes white spots bigger


"""
Verify if the region inside a contour is a table
If it is a table, returns the bounding rect
and the table joints. Else return None.
"""
MIN_TABLE_AREA = 50  # min table area to be considered a table
EPSILON = 3  # epsilon value for contour approximation


def verify_table(contour, intersections):
    area = cv.contourArea(contour)

    if (area < MIN_TABLE_AREA):
        return (None, None)

    # approxPolyDP approximates a polygonal curve within the specified precision
    curve = cv.approxPolyDP(contour, EPSILON, True)

    # boundingRect calculates the bounding rectangle of a point set (eg. a curve)
    rect = cv.boundingRect(curve)  # format of each rect: x, y, w, h

    # Finds the number of joints in each region of interest (ROI)
    # Format is in row-column order (as finding the ROI involves numpy arrays)
    # format: image_mat[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w]
    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    (possible_table_joints, _) = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Determines the number of table joints in the image
    # If less than 5 table joints, then the image
    # is likely not a table
    if len(possible_table_joints) < 5:
        return (None, None)

    return rect, possible_table_joints
