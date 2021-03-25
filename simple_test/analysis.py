import json
import os
import argparse


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


def get_parser():
    parser = argparse.ArgumentParser(description="test 1000")
    parser.add_argument(
        "--json_dir",
        default="./1000",
        help="dir of json files",
    )
    parser.add_argument(
        "--thresh",
        default=0.5,
        type=float,
        help="thresh",
    )
    return parser


def judge(json_data, thresh) -> tuple:
    """

    :param json_data:
    :param thresh:
    :return: 0-correct, 1-kill -1-skip
    """
    label = len(json_data["object"]) == 0

    result = data["pred_result"]["result"]
    pred = True
    for r in result:
        if r["score"] > thresh:
            pred = False
            break

    return label, pred


if __name__ == "__main__":
    args = get_parser().parse_args()

    result_dir = args.json_dir
    thresh = args.thresh
    json_paths = [
        os.path.join(result_dir, filename) for filename in os.listdir(result_dir)
        if filename.endswith("json")
    ]

    kill = {

    }

    skip = {

    }

    total_ture = {

    }

    total_false = {

    }

    total = {

    }

    for path in json_paths:
        with open(path) as fp:
            data = json.load(fp)
        l, p = judge(data, thresh)

        image_type = get_image_type(data["path"])

        if l is True:
            total_ture[image_type] = total_ture.get(image_type, 0) + 1
            if p is False:
                kill[image_type] = kill.get(image_type, 0) + 1
        else:
            total_false[image_type] = total_false.get(image_type, 0) + 1
            if p is True:
                skip[image_type] = skip.get(image_type, 0) + 1
        total[image_type] = total.get(image_type, 0) + 1

    for key in total.keys():
        print(f"**{key}-{thresh}")
        print(
            f"检: {skip.get(key, 0)} / {total_false.get(key, 0)} = {skip.get(key, 0) / (total_false.get(key, 0) + 1)}")
        print(
            f"误杀: {kill.get(key, 0)} / {total_ture.get(key, 0)} = {kill.get(key, 0) / (total_ture.get(key, 0) + 1)}")
