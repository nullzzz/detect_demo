import os
import random

import numpy as np
from PIL import Image

from utils.to_json import Annotation


class ImageFaker:
    def __init__(self, fore_dir: str, back_dir: str):
        self.fore_dir = fore_dir
        self.back_dir = back_dir

        self.fore_paths = os.listdir(fore_dir)
        self.back_paths = os.listdir(back_dir)

    @staticmethod
    def random_transform(image: Image.Image) -> np.ndarray:
        # rotate
        if random.random() > 0.7:
            image = image.transpose(Image.ROTATE_90)

        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # resize
        f1, f2 = 1 + random.random() * 4, 1 + random.random() * 4
        w, h = image.size
        nw = int(w * f1)
        nh = int(h * f2)
        image = image.resize((nw, nh))
        image = image.point(lambda p: p * random.choice((0.7, 0.8, 0.9, 1.1, 1.2, 1.3)))

        return np.array(image)

    def get_fore_image(self) -> tuple:
        name = random.choice(self.fore_paths)
        path = os.path.join(self.fore_dir, name)
        label = int(name[0])
        pil_image = Image.open(path)

        image = self.random_transform(pil_image)
        return image, label

    def get_back_image(self) -> np.ndarray:
        name = random.choice(self.back_paths)
        path = os.path.join(self.back_dir, name)
        pil_image:Image.Image = Image.open(path)
        if random.random() > 0.5:
            pil_image = pil_image.transpose(Image.ROTATE_90)

        image = np.array(pil_image)
        return image


class CFaker(ImageFaker):
    def __init__(self, **kwargs):
        super(CFaker, self).__init__(**kwargs)

    def generate(self, **kwargs):
        pass


mp = {
    0: "hair",
    1: "spot",
    2: "dirt",
    3: "scratch"
}
img_dir = "buffer"


def generate(count_of_defect: int = 5, **kwargs):
    back_image = faker.get_back_image()

    objs = []
    for _ in range(count_of_defect):
        fore_image, label = faker.get_fore_image()
        fore_image[fore_image < 100] = 0

        fh, fw = fore_image.shape[:2]
        bh, bw = back_image.shape[:2]
        if bw - fw < 0 or bh - fh < 0:
            continue

        x = random.randint(0, bw - fw)
        y = random.randint(0, bh - fh)

        for row in range(fh):
            for col in range(fw):
                if all(fore_image[row, col, :]) != 0:
                    back_image[row + y, col + x, :] = fore_image[row, col, :]
        objs.append({
            "label": mp[label],
            "box": [x, y, x + fw, y + fh]
        })
    path = f"{kwargs.get('path')}"
    ann.to_json(
        path=path,
        height=back_image.shape[0],
        width=back_image.shape[1],
        object=objs
    )
    image = Image.fromarray(back_image.astype(np.uint8)).convert('RGB')
    image.save(f"{img_dir}/{path}.jpg")
    return back_image


if __name__ == "__main__":
    ann = Annotation("buffer")

    faker = ImageFaker("templates/fore", "templates/back")
    for i in range(4000):
        cnt = random.randint(5, 12)
        generate(cnt, path=i)
        print(f"image {i} with {cnt} defects.")
