import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io
from detect import plot_dominant_colors

from object import get_contours, find_polygon


def show_objects(image, contours, poly_id):
    buff = np.zeros_like(image)
    for i in range(len(contours)):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, i, (255, 255, 255), -1)
        buff[mask == 255] = image[mask == 255]

        x, y, w, h = cv2.boundingRect(contours[i])
        if i == poly_id:
            cv2.rectangle(buff, (x, y), (x + w, y + h), (255, 0, 0), 5)
        else:
            cv2.rectangle(buff, (x, y), (x + w, y + h), (0, 255, 0), 5)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax2.imshow(buff)
    plt.show()


def check_image(image_path):
    image = io.imread(image_path)
    contours = get_contours(image)
    background_image = io.imread("../objects/0.jpg")
    poly_id = find_polygon(image, contours, background_image)
    show_objects(image, contours, poly_id)


if __name__ == "__main__":
    for i in range(1, 21):
        check_image("../tests/" + str(i) + ".jpg")

