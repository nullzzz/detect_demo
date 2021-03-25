import cv2
import numpy as np

"""
裁剪图像黑边
"""


def trim(image_path, **kwargs):
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = image_path
    else:
        raise TypeError()

    x1, y1 = 0, 0
    y2, x2 = img.shape[:2]
    x2 -= 1
    y2 -= 1

    while x1 < x2:
        if img[:, x1, :].sum() < 10:
            x1 += 1
        else:
            break
    while x2 > x1:
        if img[:, x2, :].sum() < 10:
            x2 -= 1
        else:
            x2 += 1
            break

    while y1 < y2:
        if img[y1, :, :].sum() < 10:
            y1 += 1
        else:
            break

    while y2 > y1:
        if img[y2, :, :].sum() < 10:
            y2 -= 1
        else:
            y2 += 1
            break

    if kwargs.get("image_need", False):
        return x1, y1, x2, y2, img[y1:y2, x1:x2, :]

    return x1, y1, x2, y2, None
