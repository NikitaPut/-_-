from core.modelling import registry
from .mnetv2 import MNETV2
from .yolo_s import YOLOs
from .csp_custom import CSP

__all__ = ['build_backcone', 'MNETV2', 'YOLOs', 'CSP']

def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED, cfg.MODEL.BACKBONE.FREEZE)