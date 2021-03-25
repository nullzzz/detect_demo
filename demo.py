# 引入包
import cv2

from detect.predictor import Predictor

# 创建一个预测对象
predictor = Predictor(gpu=True, top=3, thresh=0.5, model_dir="../model", configs_dir="../configs", trim=True)
# gpu: 是否使用gpu预测， 默认为True
# top: 缺陷检测数量，默认3个, 设为0全部检出显示
# thresh: 缺陷阈值，默认为0.5
# model_dir: 模型存放路径
# configs_dir: 配置文件存放路径
# trim: 裁剪黑边，默认为True

# 检测
# 定义一张图像
# 输入图像的文件路径
image1: str = "path/to/image"
# 输入图像的矩阵对象，opencv读入
image2 = cv2.imread("path/to/image")
# 输入预测器
result: list = predictor.predict(image1, station="C", display=True, thresh=0.6)
# station: 工位信息，必选参数，字符串形式，"A", "C", "D"
# display: 是否在预测过程中展示预测结果，可选参数，默认为False
# thresh: 缺陷阈值，可选参数，默认等于预测器定义时声明阈值

# result返回结果，列表类型
# [detect1, detect2, ……]
# 列表元素为字典类型，表示一个缺陷的检测结果
# {"box": [x1, y1, x2, y2], "score": score, "class": ""}
# box表示一个4元素列表，每个元素为整型
# score表示一个浮点数
# class表示一个字符串，为缺陷类型：hair、black spot、 dirt……

# 其他接口
# 保存预测图像
predictor.save("path/to/save")
# 获取标记框，列表形式，每个列表元素形式等同于result中的box
boxes: list = predictor.get_boxes()
