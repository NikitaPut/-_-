from torch import nn
from core.modelling.backbone import build_backbone
from core.modelling.head import build_head


class MulticlassSegmentator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

    def export_rebuild(self, target):
        self.backbone.export_rebuild(target)
        self.head.export_rebuild(target)

    def forward(self, images):
        features = self.backbone(images)
        return self.head(features)