import json
import os

from detect.predictor import Predictor

predictor = Predictor(thresh=0.2)

index = 0


def predict(image_path, station: str):
    global index
    result = predictor.predict(image_path, station, display=True)
    predictor.save(f"./1000/{station}-{index}.jpg")
    index += 1
    return result


def get_image_type(image_name):
    if "A" in image_name:
        return "A"
    elif "D" in image_name:
        return "D"
    elif "B1" in image_name or "B3" in image_name or "B7" in image_name or "B8" in image_name:
        return "PN"
    elif "B2" in image_name or "B4" in image_name or "B5" in image_name or "B6" in image_name:
        return "PW"
    else:
        return "C"


data_dir = r'/media/asus/E/test_dataset1000'
jpg_dir = os.path.join(data_dir, "image")
ann_dir = os.path.join(data_dir, "annotation")

for ann in os.listdir(ann_dir):
    ann_path = os.path.join(ann_dir, ann)
    jpg_path = os.path.join(jpg_dir, ann.replace("json", "jpg"))
    image_type = get_image_type(ann)

    result = predict(jpg_path, station=image_type)

    with open(ann_path) as f:
        data = json.load(f)

    data["pred_result"] = {
        "date": "0317",
        "result": result
    }

    with open(f"./1000/{ann}", "w") as f:
        json.dump(data, f, indent=4)
    print(ann)