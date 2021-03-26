# detect demo

***

## detect

+ predictor: 预测类

## fake data

生成数据图像

+ templates: 前景缺陷及背景图像，bmp格式

## simple test

简单测试脚本

## train

训练脚本

## utils

各类处理脚本

+ analyse: 检测结果分析
+ to coco: 转换数据集
+ to json: 定义单图像标注格式

# Todo list
***
## 关于数据

+ **A/D面非缺陷前景提取**
+ **P面数据生成**
+ 划痕、脏污前景缺陷怎么跟背景更好融合
+ 一批高质量图像供测试

## 关于训练和检测

+ detectron2 0.4版本运行错误
+ 摆脱detectron2依赖的检测

# 其他说明
***
## 数据标签

+ 0: hair
+ 1: spot
+ 2: dirt
+ 3: scratch