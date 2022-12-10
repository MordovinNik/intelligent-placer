import numpy as np
from skimage import io
import cv2
from object import Object


class Placer:

    def __init__(self, background_image_path, dataset_images_path):
        self.background_image = io.imread(background_image_path)
        self.dataset_images_path = [io.imread(img) for img in dataset_images_path]
        self.image = None
        self.polygon = None
        self.objects = None

    def draw_objects(self):
        img = np.zeros_like(self.image)
        for obj in self.objects:
            obj.paint(img)
            obj.paint_bounding_box(img, (0, 255, 0), 5)
        self.polygon.paint(img)
        self.polygon.paint_bounding_box(img, (255, 0, 0), 5)
        return img

    def load_image(self, image_path: str) -> int:
        self.image = io.imread(image_path)
        self.image = cv2.resize(self.image, [900, 1200])
        self.objects, self.polygon = Object.find_objects_on_image(self.image, self.background_image)
        return 0

    def place_objects(self, n, iterations):
        # run a collision calculation cycle, and count total intersection area
        # we will consider the problem solved if S_int < n*S
        # where S_int - total area of intersections of objects
        # S - total area of objects

        # swap objects if needed

        # if the algorithm does not provide a solution in the allotted number of iterations,
        # then the problem probably has no solution

        s = sum([cv2.contourArea(obj.contour) for obj in self.objects])
        if s < cv2.contourArea(self.polygon.contour):
            return True
        else:
            return False


