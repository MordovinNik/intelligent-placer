import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from skimage import io
from detect import plot_dominant_colors
import torchvision.ops.boxes as bops

from object import get_contours, find_polygon


def get_intersection_contour(cont1, cont2):
    x1, y1, w1, h1 = cv2.boundingRect(cont1)
    x2, y2, w2, h2 = cv2.boundingRect(cont2)

    area1 = cv2.contourArea(cont1)
    area2 = cv2.contourArea(cont2)

    box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
    box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)

    iou = bops.box_iou(box1, box2)

    if iou == 0:
        return None, None
    if area1 >= area2:
        h = h2
        w = w2
        x = x2
        y = y2
    else:
        h = h1
        w = w1
        x = x1
        y = y1
    contour_canvas1 = np.zeros((h, w), dtype='uint8')
    contour_canvas2 = np.zeros((h, w), dtype='uint8')
    cv2.drawContours(contour_canvas1, [cont1], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))
    cv2.drawContours(contour_canvas2, [cont2], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))
    contour_canvas1[contour_canvas2 == 0] = 0
    return contour_canvas1, [x, y]


def collide_objects(cont1, cont2, center1, center2, intersection):
    center = find_center(intersection)
    a = center1 - center
    b = center2 - center
    mass = np.count_nonzero(intersection)
    l = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
    angle = np.arccos((a[0] * b[0] + a[1] * b[1]) / (l(a) * l(b)))
    # передвинуть объекты
    c = np.int_(a * np.cos(angle) * mass / cv2.contourArea(cont1)/2)
    d = np.int_(b * np.cos(angle) * mass / cv2.contourArea(cont2)/2)
    print(f"a = {c}\nb = {d}")
    cont1 += c
    cont2 += d


def place_objects(image, contours, poly_id):
    # расположить объекты на холсте
    poly = contours[poly_id]
    cont = []
    for i in range(len(contours)):
        if i != poly_id:
            cont.append(contours[i])

    # найти центры объектов
    centers = [find_center(i) for i in cont]
    # найти пересечения
    for i in range(len(cont)):
        for j in range(i + 1, len(cont)):
            intersection, coords = get_intersection_contour(cont[i], cont[j])
            if intersection is None:
                continue
            # найти центры пересечений
            print(f"sum = {np.count_nonzero(intersection)}")
            # оттолкнуть объекты друг от друга
            collide_objects(cont[i], cont[j], centers[i], centers[j], intersection)


def find_center(contour):
    moments = cv2.moments(contour)
    return np.array(np.uint([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]))


def show_objects(image, contours, poly_id):
    buff = np.zeros_like(image)
    contours[0] = contours[0] - [-100, 40]
    place_objects(image, contours, poly_id)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(len(contours)):
        mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        center = find_center(contours[i])
        print(center)
        cv2.drawContours(mask, contours, i, 255, -1)
        # buff[mask == 255] = image[mask == 255]
        cv2.copyTo(image, mask, buff)
        cv2.circle(buff, center, 5, [255, 0, 0], 3)

        x, y, w, h = cv2.boundingRect(contours[i])
        if i == poly_id:
            cv2.rectangle(buff, (x, y), (x + w, y + h), (255, 0, 0), 5)
        else:
            cv2.rectangle(buff, (x, y), (x + w, y + h), (0, 255, 0), 5)


    ax1.imshow(image)
    ax2.imshow(buff)
    plt.show()


def check_image(image_path):
    image = io.imread(image_path)
    image = cv2.resize(image, [900, 1200])
    contours = get_contours(image)
    background_image = io.imread("../objects/0.jpg")
    poly_id = find_polygon(image, contours, background_image)
    show_objects(image, contours, poly_id)


if __name__ == "__main__":
    for i in range(1, 2):
        check_image("../tests/" + str(i) + ".jpg")
