import warnings

import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt

MIN_CONTOUR_AREA = 2000
MAX_COLOR_DIFFERENCE = 25
DEBUG_CONTOURS = False
DEBUG_POLYGON_FIND = False


def get_contours(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image_gauss = cv2.GaussianBlur(image_gray, (5, 5), 0)
    ret, image_threshold = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    image_dilated = cv2.dilate(image_threshold, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ext_contours = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > MIN_CONTOUR_AREA:
            ext_contours.append(contours[i])
            print(f"contour area: {cv2.contourArea(contours[i])}")

    if DEBUG_CONTOURS:
        result = np.zeros((image_dilated.shape[0], image_dilated.shape[1], 3), dtype=np.uint8)
        for i in range(len(ext_contours)):
            color = (randint(0, 256), randint(0, 256), randint(0, 256))
            cv2.drawContours(result, ext_contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image_dilated)
        ax2.imshow(result)
        plt.show()

    return ext_contours


def find_polygon(image, contours, background_image):
    # to do: replace by dominant color comparison???
    # find background average color and normalize image light
    average = background_image.mean(axis=0).mean(axis=0)
    mod = average.mean() / average
    average *= mod
    image_modified = np.int_(image * mod)
    image_modified[image_modified > 255] = 255

    # find object with most similar average color
    min = 255
    min_i = -1
    max_diff = 0
    sum_mask = np.ndarray((image_modified.shape[0], image_modified.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cut_object = np.ones((image_modified.shape[0], image_modified.shape[1], 3), dtype=np.uint8)
        mask = np.zeros_like(image_modified)
        cv2.drawContours(mask, contours, i, (255, 255, 255), -1)
        cut_object[mask == 255] = image_modified[mask == 255]
        sum_mask[mask == 255] = 255
        boo = mask == 255

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg = np.nanmean(np.nanmean(image_modified, axis=0, where=mask == 255), axis=0)
            diff = np.linalg.norm(avg - average)
            max_diff = max(abs(avg - average))
            if min > diff:
                min_i = i
                min = diff

    if max_diff > MAX_COLOR_DIFFERENCE:
        min_i = -1

    if DEBUG_POLYGON_FIND:
        print("FUNCTION:find_polygon")
        print(f"mod = {mod}")
        print(f"background average: {average}")
        print(f"contours count:{len(contours)}, min_index = {min_i}, min = {min}, max_diff = {max_diff}")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(sum_mask)
        ax2.imshow(image_modified)
        plt.show()
    return min_i


