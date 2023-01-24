from core.modelling import registry
from .DeepLabV3 import DeepLabV3_Custom

__all__ = ['build_head', 'DeepLabV3_Custom']

def build_head(cfg):
    return registry.HEADS[cfg.MODEL.HEAD.NAME](cfg, cfg.MODEL.HEAD.PRETRAINED, cfg.MODEL.HEAD.FREEZE)