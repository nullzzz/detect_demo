import random

import cv2
import numpy as np

from fake_data.fake_image_master import ImageFaker
from utils.to_json import Annotation


def fake_image(fore, back, count: int = 1, transform=None, region: list = None, **kwargs) -> np.ndarray:
    """
    对给定前景缺陷fore, 随机贴到背景back
    :param fore: 图片路径或opencv图像
    :param back: 同上
    :param count: 生成的缺陷数量
    :param transform: 一个或多个函数，对前景缺陷进行随机变化
    :param region: 前景贴的区域
    :return:
    """
    if isinstance(fore, str):
        fore_image: np.ndarray = cv2.imread(fore)
    elif isinstance(fore, np.ndarray):
        fore_image: np.ndarray = fore
    else:
        raise TypeError()

    if isinstance(back, str):
        back_image: np.ndarray = cv2.imread(back)
    elif isinstance(back, np.ndarray):
        back_image: np.ndarray = back
    else:
        raise TypeError()

    if transform is not None:
        if isinstance(transform, (list, tuple)):
            trans = random.choice(transform)
        else:
            trans = transform
        fore_image = trans(fore_image)

    fore_image[fore_image < 80] = 0
    # todo 1, 限定前景贴上的区域，裁剪下region区域，按当前方法paste，然后region区域paste回原背景
    # todo 2, region支持多区域
    if region is not None:
        x1, y1, x2, y2 = region
        region_area = (x2 - x1) * (y2 - y1)
        assert region_area > fore.shape[0] * fore.shape[1], "region too small to paste fore image."

    fh, fw = fore_image.shape[:2]
    bh, bw = back_image.shape[:2]

    objs = []
    for _ in range(count):
        x = random.randint(0, bw - fw)
        y = random.randint(0, bh - fh)

        for row in range(fh):
            for col in range(fw):
                if all(fore_image[row, col, :]) != 0:
                    back_image[row + y, col + x, :] = fore_image[row, col, :]
        objs.append({
            "label": "black spot",
            "box": [x, y, x + fw, y + fh]
        })
    path = f"black_spot_{kwargs.get('path')}"
    ann.to_json(
        path=path,
        height=back_image.shape[0],
        width=back_image.shape[1],
        object=objs
    )
    cv2.imwrite(f"{path}.jpg", back_image)
    return back_image


if __name__ == "__main__":
    ann = Annotation("../utils/fake_image/buffer")
    faker = ImageFaker("templates/hair", "templates/background")
    for _ in range(10):
        fakes = fake_image(faker.get, back_path, random.randint(1, 5), path=f"{_}")
