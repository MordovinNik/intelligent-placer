import warnings
import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt

MIN_CONTOUR_AREA = 2000
MAX_COLOR_DIFFERENCE = 25
DEBUG_CONTOURS = False
DEBUG_POLYGON_FIND = False


def find_center(contour):
    moments = cv2.moments(contour)
    return np.array(np.uint([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]))


class Object:

    def __init__(self, image, contour):
        self.center = find_center(contour)
        self.image = image
        self.contour = contour
        self.angle = 0

    @staticmethod
    def find_contours_on_image(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, image_threshold = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        image_dilated = cv2.dilate(image_threshold, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ext_contours = []
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > MIN_CONTOUR_AREA:
                ext_contours.append(contours[i])

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

    @staticmethod
    def find_polygon_index(image, contours, background_image):
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
            print("FUNCTION: find_polygon_index")
            print(f"mod = {mod}")
            print(f"background average: {average}")
            print(f"contours count:{len(contours)}, min_index = {min_i}, min = {min}, max_diff = {max_diff}")
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(sum_mask)
            ax2.imshow(image_modified)
            plt.show()

        return min_i

    @staticmethod
    def find_objects_on_image(image, background_image):
        contours = Object.find_contours_on_image(image)
        polygon_index = Object.find_polygon_index(image, contours, background_image)
        polygon = Object(image, contours.pop(polygon_index))
        found_objects = [Object(image, contour) for contour in contours]
        return found_objects, polygon

    def paint(self, buffer):
        mask = np.zeros_like(buffer)
        cv2.drawContours(mask, [self.contour], 0, (255, 255, 255), -1)
        buffer[mask == 255] = self.image[mask == 255]

    def paint_bounding_box(self, buffer, color, width):
        x, y, w, h = cv2.boundingRect(self.contour)
        cv2.rectangle(buffer, (x, y), (x + w, y + h), color, width)

    @staticmethod
    def collide_objects(obj1, obj2, force_mult=1):
        # find intersection between two objects
        # and push them apart with some rotation and force depending on the intersection area
        pass

    @staticmethod
    def collide_with_borders(obj, polygon, force_mult=1):
        # find the part of the object protruding beyond the border
        # and push the object away from the border with force depending on protruding area
        pass
