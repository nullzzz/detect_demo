import json

"""
a 脏污
b 黑点
c 异色
e 拉伤
g 凹坑
i 毛丝
j 其他
"""


def pythagorean_theorem(p1, p2) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def label_me(ann_path):
    label_classes = {
        "a": "dirt",
        "b": "spot",
        "c": "dirt(harmonia)",
        "e": "scratch",
        "g": "pit",
        "i": "hair",
        "j": "other",

    }
    with open(ann_path) as ann_file:
        ann_dict = json.loads(ann_file.read())

    shapes = ann_dict["shapes"]

    objects = []
    for shape in shapes:
        points = shape.get("points")
        shape_type = shape.get("shape_type")
        if shape_type == "circle":
            radius = pythagorean_theorem(points[0], points[1])
            x1 = points[0][0] - radius
            y1 = points[0][1] - radius
            x2 = points[0][0] + radius
            y2 = points[0][1] + radius
        elif shape_type == "polygon" or shape_type == "rectangle" or shape_type == "linestrip":
            x1 = float("inf")
            x2 = 0
            y1 = float("inf")
            y2 = 0
            for (x, y) in points:
                x1 = min(x1, x)
                x2 = max(x2, x)
                y1 = min(y1, y)
                y2 = max(y2, y)
        else:
            continue
        objects.append(dict(
            label=label_classes.get(shape.get("label")),
            box=[int(x1), int(y1), int(x2), int(y2)]
        ))
    return objects
