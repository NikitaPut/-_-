from .transforms2 import *


class TransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels=None, masks=None):
        for t in self.transforms:
            images, labels, masks = t(images, labels, masks)

        return images, labels, masks


def build_transforms(image_size, is_train:bool=True, to_tensor:bool=True):
    if is_train:
        transform = [
            # RandomCrop(min_crop=0.7, probabilty=0.5),
            Resize(image_size),
            RandomRotate(-90.0, 90.0),
            ConvertFromInts(),
            # RandomGamma(probabilty=0.5),
            # RandomHue(probabilty=0.5),
            RandomMirror(horizont_prob=0.5, probabilty=0.5),
            Clip()
        ]
    else:
        transform = [
            Resize(image_size),
            ConvertFromInts()
        ]

    if to_tensor:
        transform = transform + [Normalize(), ToTensor(norm_label=True, norm_mask=True)]

    transform = TransformCompose(transform)
    return transform