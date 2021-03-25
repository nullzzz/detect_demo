import json
import os
import re
import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt

class BoundingBox:
    def __init__(self, **kwargs):
        self.box = kwargs.get("box")
        self.score = kwargs.get("score")
        self.area = (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    def to_json(self) -> dict:
        jsn: dict = {
            "score": self.score,
            "box": self.box,
            "area": self.area
        }
        return jsn


class Result:
    def __init__(self, **kwargs):
        self.image = kwargs.pop("image", "")
        self.label = kwargs.pop("label", None)
        self.boxes: list = []

    def parse_from_out(self, out):
        for idx in range(len(out["instances"])):
            box = out["instances"][idx].get("pred_boxes")
            scores = out["instances"][idx].get("scores")
            score = scores[0].item()
            x1, y1, x2, y2 = int(box.tensor[0, 0]), int(box.tensor[:, 1]), \
                             int(box.tensor[:, 2]), int(box.tensor[:, 3])
            self.boxes.append(BoundingBox(box=[x1, y1, x2, y2], score=score))

    def set_label(self, label: bool):
        self.label = label

    def set_image_path(self, path):
        self.image = path

    def to_json(self) -> dict:
        jsn: dict = {
            "image_path": self.image,
            "label": self.label,
            "boxes": [box.to_json() for box in self.boxes]
        }

        return jsn

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, item):
        return self.boxes[item]

    def __str__(self):
        return f"image: {self.image} \n" \
               f"label: {self.label} \n" \
               f"count of bbox: {len(self.boxes)}"

    def judge(self, score: float, area: int = 0):
        for bbox in self.boxes:
            if bbox.score > score and bbox.area > area:
                return True
        return False


class Analyser:
    def __init__(self, result_file_dir: str = None, **kwargs):
        self.save_dir = kwargs.get("save_dir", "../")

        self.data = []
        if result_file_dir is not None:
            for jsp in os.listdir(result_file_dir):
                with open(os.path.join(result_file_dir, jsp)) as f:
                    self.data.append(json.load(f))

    def load_from_json(self, json_path):
        with open(json_path) as jf:
            self.data = json.load(jf)

    def update(self, image_path: str, label, out):
        result = Result()
        result.parse_from_out(out)
        result.set_label(label=label)
        result.set_image_path(os.path.basename(image_path))

        fp = open(os.path.join(self.save_dir, os.path.basename(image_path) + ".json"), "w")
        print(os.path.join(self.save_dir, os.path.basename(image_path) + ".json"))
        print(result.to_json())
        json.dump(result.to_json(), fp)
        fp.close()

    def analyse_excel(self, thresh: float):
        dd = {"a": [1] * 86, "cb": [1] * 86, "cl": [1] * 86, "cr": [1] * 86, "pn": [1] * 86, "pw": [1] * 86,
              "d": [1] * 86}
        df = pd.DataFrame(dd)

        with open("./map.json") as fp:
            f = json.load(fp)
        for i in self.data:
            image_name = i["image_path"]
            key = image_name.split('.')[0]
            index = f[key]
            # print(index)
            i_type = self.get_image_type(i["image_path"])

            boxes = i["boxes"]
            for box in boxes:
                if box["score"] >= thresh:
                    df.loc[int(index), i_type.lower()] = 2
        return df

        print(df)

    def analyse(self, thresh: float):
        assert 0 < thresh < 1.0, "Thresh must between 0~1."
        judge_type = ["正常", "漏检", "过杀"]

        list_type_all = {"a": 0, "pw": 0, "pn": 0, "cb": 0, "cl": 0, "cr": 0, "d": 0, }
        list_type_gs = {"a": 0, "pw": 0, "pn": 0, "cb": 0, "cl": 0, "cr": 0, "d": 0, }
        gs = [0.0] * 7
        list_type_lj = {"a": 0, "pw": 0, "pn": 0, "cb": 0, "cl": 0, "cr": 0, "d": 0, }
        lj = [0.0] * 7

        for i in self.data:
            i_type = self.get_image_type(i["image_path"])
            i_type = i_type.lower()

            list_type_all[i_type] = list_type_all[i_type] + 1

            if self.judge(i, thresh) == judge_type[1]:
                list_type_lj[i_type] = list_type_lj[i_type] + 1
            elif self.judge(i, thresh) == judge_type[2]:
                list_type_gs[i_type] = list_type_gs[i_type] + 1

        for index, key in enumerate(list_type_all.keys()):
            gs[index] = list_type_gs[key] / list_type_all[key]
            lj[index] = list_type_lj[key] / list_type_all[key]

        print("tresh:", thresh)
        print("all:", list_type_all)
        print("过杀：", list_type_gs)
        print("漏检：", list_type_lj)

    def judge(self, data: dict, thresh: float):
        image_path = data["image_path"]

        label = data["label"]

        boxes = data["boxes"]
        num = 0
        for box in boxes:
            if thresh <= box["score"]:
                num += 1
        if num >= 0 and label != "False":
            return "过杀"
        elif (num == 0 and label == "True") or (num >= 0 and label == "False"):
            return "正常"
        else:
            return "漏检"

    def get_image_type(self, filename: str):
        """
        根据文件名判断哪个面
        :param filename:
        :return:
        """
        if "A" in filename or "ST2_PH0" in filename:
            return "A"
        elif "D" in filename or "ST2_PH1" in filename:
            return "D"
        elif "B1" in filename or "B3" in filename or "B7" in filename or "B8" in filename or "ST1_PH0" in filename or "ST1_PH1" in filename:
            return "PN"
        elif "B2" in filename or "B4" in filename or "B5" in filename or "B6" in filename or "ST1_PH2" in filename or "ST1_PH3" in filename or "ST1_PH4" in filename or "ST1_PH5" in filename:
            return "PW"
        elif "C1_1" in filename or "C3_1" in filename or "ST0_PH2" in filename or "ST0_PH6" in filename or "C1" in filename or "C3" in filename:
            return "CB"
        elif "C2_1" in filename or "C4_1" in filename or "ST0_PH1" in filename or "ST0_PH3" in filename or "ST0_PH5" in filename or "ST0_PH7" in filename or "C2" in filename or "C4" in filename:
            return "CL"
        elif "CR" in filename or "ST0_PH0" in filename or "ST0_PH4" in filename:
            return "CR"


# Wazkbcldlm

# path = r'/media/asus/E/test_data3/product.xls'
# book = xlrd.open_workbook(path)
# names = book.sheet_names()
# sheet = book.sheet_by_name(names[0])
#
# ans = Analyser('./res-1036')
# dict1 = {
#     "C大":2,
#     "C小":3,
#     "C弧":4,
#     "pin宽":6,
#     "pin窄":5,
#     "A":1,
#     "D":7,
# }
#
# years = ['A', 'C', 'Cl', 'Cr', 'pinn', "pinw", "D"]
# fig = plt.figure(figsize=(10,10))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# for th in np.arange(0.5,1,0.1):
#     df = ans.analyse_excel(th)
#
#     gs = [0] * 7
#     lj = [0] * 7
#
#     for col in range(0,7):
#         for row in range(0,86):
#             if sheet.cell_value(row+1,col+1) != sheet.cell_value(2,7):
#                 if df.iloc[row,col] == 1:
#                     lj[col] = lj[col] + 1
#             else:
#                 if df.iloc[row,col] == 2:
#                     gs[col] = gs[col] + 1
#     for i in range(7):
#         gs[i] = round(gs[i] / 85, 2)
#         lj[i] = round(lj[i] / 85, 2)
#     print("阈值：",th)
#
#     print("过杀：",gs)
#     plt.subplot(211)
#     plt.plot(years, gs, '.-', label='thresh' + str(round(th,2)))
#     print("漏检：",lj)
#     plt.subplot(212)
#     plt.plot(years, lj, '.-', label='thresh' + str(round(th,2)))
#
# plt.subplot(211)
# plt.title("over")
# plt.legend()
# plt.subplot(212)
# plt.title("miss")
# plt.legend()
# plt.show()  # 显示图形








