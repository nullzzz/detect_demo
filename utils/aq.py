import json

for i in range(1000):
    fn = f"fake_image/spot/black_spot_{i}.json"
    with open(fn) as f:
        data = json.load(f)
        for obj in data["object"]:
            obj["label"] = "spot"

    with open(fn, "w") as f:
        json.dump(data, f, indent=4)
