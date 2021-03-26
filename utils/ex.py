import torch
from detectron2.config import get_cfg
from detectron2.export import scripting_with_instances
from detectron2.modeling import build_model
from detectron2.structures import Boxes

cfg = get_cfg()
cfg.merge_from_file("configs/faster_rcnn_R_50_FPN_1x.yaml")

cfg.MODEL.WEIGHTS = "model/model_a.pth"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    1
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.DEVICE = "cpu"
model = build_model(cfg)
model.eval()
fields = {"pred_boxes": Boxes, "scores": torch.Tensor, "pred_classes": torch.Tensor, "proposal_boxes": Boxes,
          "objectness_logits": torch.Tensor}
torchscipt_model = scripting_with_instances(model, fields)

torch.jit.save(torchscipt_model, "model/jit_a.pth")

# 提取缺陷
