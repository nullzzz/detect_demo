import random

from fake_data.fake_image_master import PinFaker
from utils.annotation import Annotation


img_dir = "buffer"

if __name__ == "__main__":
    ann = Annotation("buffer")

    faker = PinFaker(fore_dir="templates/fore", back_dir="templates/back/pw", output_dir=img_dir, ann=ann)
    for i in range(20):
        cnt = random.randint(5, 12)
        faker.generate(cnt, path=i)
        print(f"image {i} with {cnt} defects.")
