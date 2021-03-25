import datetime
import json
import os


def to_coco(ann_dir, output_dir):
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            {
                "supercategory": None,
                "id": 1,
                "name": "hair",
            },
            {
                "supercategory": None,
                "id": 2,
                "name": "spot",
            },
            {
                "supercategory": None,
                "id": 3,
                "name": "dirt",
            },
            {
                "supercategory": None,
                "id": 4,
                "name": "scratch",
            }
        ],
    )

    class_name_to_id = {
        "hair": 1,
        "spot": 2,
        "dirt": 3,
        "scratch": 4
    }

    out_ann_file = os.path.join(output_dir, "annotations.json")
    ann_files = [os.path.join(ann_dir, filename) for filename in os.listdir(ann_dir) if filename.endswith("json")]

    for image_id, filename in enumerate(ann_files):
        print("Generating dataset from:", filename)

        with open(filename) as f:
            label_file: dict = json.load(f)

        h = label_file.get("height")
        w = label_file.get("width")
        fn = label_file.get("path")

        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=fn,
                height=h,
                width=w,
                date_captured=None,
                id=image_id,
            )
        )

        for obj in label_file.get("object"):
            label = obj["label"]

            x1, y1, x2, y2 = obj.get("box")
            area = (x2 - x1) * (y2 - y1)
            bbox = [x1, y1, x2 - x1, y2 - y1]

            if label not in class_name_to_id:
                continue
            cls_id = class_name_to_id[label]
            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=None,
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

    with open(out_ann_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    to_coco("../fake_data/buffer", "../fake_data/buffer")
