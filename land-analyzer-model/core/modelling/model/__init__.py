from .multiclass_segmentator import MulticlassSegmentator

_MODEL_META_ARCHITECTURES = {
    "MulticlassSegmentator": MulticlassSegmentator,
}

def build_model(cfg):
    meta_arch = _MODEL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)