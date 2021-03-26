import os
import random

import numpy as np
from PIL import Image

from utils.annotation import Annotation


class Defect:
    def __init__(self, image: np.ndarray, label):
        self.image = image
        self.label = label

    def color(self):
        if self.label != 1:
            self.image = self.image.point(lambda p: 0 if p < 80 else p * random.choice((0.7, 0.8, 0.9, 1.1, 1.2, 1.3)))
        else:
            self.image = self.image.point(lambda p: p * random.choice((0.7, 0.8, 0.9, 1.1, 1.2)))

    def transform(self):
        # rotate
        if random.random() > 0.7:
            self.image = self.image.transpose(Image.ROTATE_90)
        # flip
        if random.random() > 0.5:
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)

        # resize
        if self.label == 2:
            f1, f2 = 0.8 + random.random(), 0.8 + random.random()
        elif self.label == 1:
            f1 = f2 = 1 + random.random()
        else:
            f1, f2 = 1 + random.random() * 2, 1 + random.random() * 2
        w, h = self.image.size
        nw = int(w * f1)
        nh = int(h * f2)
        self.image = self.image.resize((nw, nh))

    def to_numpy(self):
        return np.array(self.image)


class ImageFaker:
    """
    0: hair
    1: spot
    2: dirt
    3: scratch
    """

    def __init__(self, fore_dir: str, back_dir: str, output_dir: str, ann):
        self.fore_dir = fore_dir
        self.back_dir = back_dir
        self.output_dir = output_dir
        self.ann = ann

        self.id_label = {
            9: "other",
            0: "hair",
            1: "spot",
            2: "dirt",
            3: "scratch"
        }

        self.fore_paths = os.listdir(fore_dir)
        self.back_paths = os.listdir(back_dir)

    @staticmethod
    def random_transform(image: Image.Image, label=None) -> np.ndarray:
        # rotate
        if random.random() > 0.7:
            image = image.transpose(Image.ROTATE_90)

        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # resize
        if label == 2:
            f1, f2 = 0.8 + random.random(), 0.8 + random.random()
        elif label == 1:
            f1 = f2 = 1
        else:
            f1, f2 = 1 + random.random() * 2, 1 + random.random() * 2
        w, h = image.size
        nw = int(w * f1)
        nh = int(h * f2)
        image = image.resize((nw, nh))
        if label != 1:
            image = image.point(lambda p: 0 if p < 120 else p * random.choice((0.7, 0.8, 0.9, 1.1, 1.2, 1.3)))
        else:
            image = image.point(lambda p: p / 255 * 150 + 80 if p != 0 else 0)

        return np.array(image)

    def get_fore_image(self) -> tuple:
        # name = random.choice([p for p in self.fore_paths if p[0] == "1"])
        name = random.choice(self.fore_paths)
        path = os.path.join(self.fore_dir, name)
        label = int(name[0])
        pil_image = Image.open(path)

        image = self.random_transform(pil_image, label)
        return image, label

    def get_back_image(self) -> np.ndarray:
        name = random.choice(self.back_paths)
        path = os.path.join(self.back_dir, name)
        pil_image: Image.Image = Image.open(path)
        if random.random() > 0.5:
            pil_image = pil_image.transpose(Image.ROTATE_90)

        image = np.array(pil_image)
        return image

    def generate(self, count_of_defect, path):
        back_image = self.get_back_image()

        objs = []
        for _ in range(count_of_defect):
            fore_image, label = self.get_fore_image()

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
                "label": self.id_label[label],
                "box": [x, y, x + fw, y + fh]
            })
        self.ann.to_json(
            path=path,
            height=back_image.shape[0],
            width=back_image.shape[1],
            object=objs
        )
        image = Image.fromarray(back_image.astype(np.uint8)).convert('RGB')
        image.save(f"{self.output_dir}/{path}.jpg")
        return back_image

    def generate_black_spot(self):
        width = random.randint(15, 50)
        height = int(width * (0.9 + random.random() / 5))

        image = np.zeros([height, width, 3], dtype=np.uint8)

        center_x, center_y = width // 2, height // 2
        radius = min(width // 2, height // 2)


class CFaker(ImageFaker):
    def __init__(self, **kwargs):
        super(CFaker, self).__init__(**kwargs)

    def generate(self, **kwargs):
        pass


class PinFaker(ImageFaker):
    def __init__(self, **kwargs):
        super(PinFaker, self).__init__(**kwargs)

    def generate(self, count_of_defect, path):
        back_image = self.get_back_image()
        objs = []
        for _ in range(count_of_defect):
            fore_image, label = self.get_fore_image()

            fh, fw = fore_image.shape[:2]
            bh, bw = back_image.shape[:2]
            if bw - fw < 0 or bh - fh < 0:
                continue

            x = random.randint(0, bw - fw)
            y = random.randint(0, bh - fh)

            back_region: np.ndarray = back_image[y:y + fh, x: x + fw, :]

            zero_count = 1 - np.count_nonzero(back_region) / (fh * fw * 3)
            if zero_count > 0.1:
                continue

            for row in range(fh):
                for col in range(fw):
                    if all(fore_image[row, col, :]) != 0:
                        back_image[row + y, col + x, :] = fore_image[row, col, :]
            objs.append({
                "label": self.id_label.get(label),
                "box": [x, y, x + fw, y + fh]
            })
        self.ann.to_json(
            path=path,
            height=back_image.shape[0],
            width=back_image.shape[1],
            object=objs
        )
        image = Image.fromarray(back_image.astype(np.uint8)).convert('RGB')
        image.save(f"{self.output_dir}/{path}.jpg")
        return back_image


if __name__ == "__main__":
    ann = Annotation("buffer/c")
    faker = PinFaker(fore_dir="templates/fore", back_dir="templates/back/c", output_dir="buffer/c", ann=ann)
    for i in range(8000):
        cnt = random.randint(5, 12)
        faker.generate(cnt, path=i)
        print(f"image {i} with {cnt} defects.")
