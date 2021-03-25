import os

import cv2
import matplotlib.pyplot as plt
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances


class Predictor:
    def __init__(self, gpu: bool = True, thresh: float = 0.5, top: int = 3, **kwargs):
        """
        初始化
        :param gpu: 是否使用gpu，True or False，默认使用
        :param thresh: 阈值，用于过滤bbox置信度，0~1，默认0.9
        :param top: 绘制bbox数量，默认3个
        :param kwargs: 其他参数
        """

        self.gpu = gpu
        if gpu is True:
            if not torch.cuda.is_available():
                print("cuda unsupported")
                self.gpu = False

        self.saved = kwargs.pop("saved", False)

        self.top = top
        # 加载detectron2模型参数
        self.model_dir = kwargs.get("MODEL_DIR", "../model")
        self.config_dir = kwargs.get("CONFIGS_DIR", "../configs")

        if categories := kwargs.get("categories"):
            self.categories = categories
        else:
            self.categories = {
                0: "dirt",
                1: "gap",
                2: "hair",
                3: "scratch",
                4: "black spot",
            }

        # 模型字典，键：工位 值：模型
        self.model_zoo: dict = {}

        # 当前预测图像提取框
        self.boxes = []
        # 当前图像置信度
        self.scores = []
        self.classes = []
        # 当前预测图像
        self.image = None
        # 候选框置信度阈值
        self.thresh = thresh

        # 按工位预加载预测模型
        for i in ('C',):
            self.load_model(i)

        self.extra_func = lambda x: x

    def predict(self, image_path: str, station: str, display: bool = False, thresh: float = None,
                label: str = "unknown",
                **kwargs) -> bool:
        """

        :param image_path: 图像
        :param station: 工位信息，字符串
        :param display: 是否绘制展示
        :param thresh: 置信度阈值
        :param label:
        :return:
        """
        self.saved = kwargs.pop("save", self.saved)

        if thresh is not None:
            self.thresh = thresh

        # 对输入进行检查
        if isinstance(image_path, str):
            if os.path.isfile(image_path):
                image = cv2.imread(image_path)
                if image is None:
                    return True
            else:
                raise ValueError("Invalid File Path")
        else:
            raise TypeError("Please input image path")

        image = self.extra_func(image)
        # 清楚上一次预测的图像信息
        self.boxes = []
        self.scores = []
        self.classes = []
        self.image = image
        # 使用对应工位模型
        model = self.model_zoo.get(station)

        out = model(image)

        instances: Instances = out.get("instances")
        fields: dict = instances.get_fields()

        boxes = fields.get("pred_boxes")
        scores = fields.get("scores")
        classes = fields.get("pred_classes")

        for b, s, c in zip(boxes, scores, classes):
            b: torch.Tensor
            s: torch.Tensor
            c: torch.Tensor

            x1, y1, x2, y2 = b.numpy()
            score = s.numpy()
            cls = c.numpy()

            self.boxes.append([x1, y1, x2, y2])
            self.scores.append(score)
            self.classes.append(cls)

        if display:
            for (x1, y1, x2, y2), score, cls in zip(self.boxes, self.scores, self.classes):
                print(64 * cls % 256)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 64 * cls % 256, 0), thickness=5)
                cv2.putText(image, f"{self.categories[cls]}-{score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2)
            plt.imshow(image)
            plt.show()

    def show(self):
        if self.image is None:
            return
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
        plt.imshow(self.image)
        plt.show()

    def save(self, path: str):
        """
        调试用
        :param path:
        :return:
        """
        assert self.image is not None, "no image, please display it before saving it."
        cv2.imwrite(path, self.image)

    def get_boxes(self):
        """
        获取候选框
        :return:
        """
        if self.boxes is []:
            print("please predict oen image before getting bounding boxes.")
        return self.boxes

    def load_model(self, station: str):
        """
        加载模型
        :param station:
        :return:
        """
        thresh = self.thresh

        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(self.config_dir, "faster_rcnn_R_50_FPN_1x.yaml"))

        if station == 'C':
            model_path = "model_c.pth"
            model = torch.jit.load("model/jit_c.pth")
        self.model_zoo[station] = model


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict(r"E:\defect_dataset\image\002064-C-68_C5#1.jpg", "C", display=True)
    predictor.save("2.jpg")
