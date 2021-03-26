import datetime
import json
import os


class Annotation:
    def __init__(self, data_dir: str = ""):
        self.data_dir = data_dir
        self.format = {
            "version": f"{datetime.date.today().month}.{datetime.date.today().day}",
            "path": "",
            "height": "",
            "width": "",
            "object": []
        }

    def to_json(self, **kwargs):
        path = kwargs.get("path")
        self.format["path"] = f"{path}.jpg"
        self.format["height"] = kwargs.get("height")
        self.format["width"] = kwargs.get("width")
        self.format["object"] = kwargs.get("object")
        with open(os.path.join(self.data_dir, f"{path}.json"), "w") as f:
            json.dump(self.format, f, indent=4)
