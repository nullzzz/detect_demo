import cv2
import numpy as np


def draw_bounding_box(image: np.ndarray, bounding_boxes):
    image = image.copy()
    for bbox in bounding_boxes:
        label = bbox.get("label", "")
        x1, x2, y1, y2 = bbox.get("box")
        assert isinstance(x1, int)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=3)
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (183, 118, 35), 2)
    return image
