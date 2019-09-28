import cv2
import numpy as np


def min_area_rect(poly):
    poly = cv2.minAreaRect(np.array(poly, 'float32'))
    if poly[2] < -45:
        poly = (poly[0], poly[1], poly[2] + 180)
    else:
        poly = (poly[0], poly[1][::-1], poly[2] + 90)
    poly = cv2.boxPoints(poly)
    return poly
