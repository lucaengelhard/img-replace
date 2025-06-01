from tinyface import Face
import numpy as np
import cv2
from typing import Union
from utils import get_img
from tqdm import tqdm


def blur_faces(img_input: Union[str, np.ndarray], faces: list[Face]):
    print()
    print(" - Blurring image")

    img = get_img(img_input)
    res = img.copy()

    height, width = img.shape[:2]
    kernel_size = round(max(width, height) / 320)

    if kernel_size % 2 == 0:
        kernel_size += 1

    print(kernel_size)

    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

    padding = 10

    print()
    print(" - Blurring faces")
    for face in tqdm(faces):
        x, y, w, h = face.bounding_box

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, width)
        y2 = min(y + h + padding, height)

        res[y1:y2, x1:x2] = blurred_img[y1:y2, x1:x2]

    return res
