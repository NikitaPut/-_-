import torch
import cv2 as cv
import numpy as np
from torch import float32


class RandomCrop(object):
    def __init__(self, min_crop:float=0.8, probabilty:float=0.5):
        self.min_crop = np.clip(min_crop, 0.0, 1.0)
        self.p = np.clip(probabilty, 0.0, 1.0)

    def __call__(self, image, labels=None, mask=None):
        if np.random.choice([0, 1], size=1, p=[1-self.p, self.p]):
            scale_x = min(self.min_crop + np.random.random() * (1.0 - self.min_crop), 1.0)
            scale_y = min(self.min_crop + np.random.random() * (1.0 - self.min_crop), 1.0)
            new_h = int(image.shape[0] * scale_y)
            new_w = int(image.shape[1] * scale_x)
            shift_y = int(np.random.random() * (image.shape[0] - new_h))
            shift_x = int(np.random.random() * (image.shape[1] - new_w))

            image = image[shift_y:new_h + shift_y, shift_x:new_w + shift_x, :]
            if labels is not None:
                for i in range(0, len(labels)):
                    labels[i] = labels[i][shift_y:new_h + shift_y, shift_x:new_w + shift_x]
            if mask is not None:
                mask = mask[shift_y:new_h + shift_y, shift_x:new_w + shift_x]

        return image, labels, mask


class Resize(object):
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, labels=None, mask=None):
        image = cv.resize(image, self.size, interpolation=cv.INTER_AREA)
        if labels is not None:
            for i in range(0, len(labels)):
                labels[i] = cv.resize(labels[i], self.size, interpolation=cv.INTER_NEAREST)
        if mask is not None:
            mask = cv.resize(mask, self.size, interpolation=cv.INTER_NEAREST)
        return image, labels, mask


class RandomRotate(object):
    def __init__(self, angle_min:float=-15, angle_max:float=15):
        self.angle_min = angle_min
        self.angle_max = angle_max
        assert self.angle_max >= self.angle_min, "angle max must be >= angle min."

    def __call__(self, image, labels=None, mask=None):
        angle = np.random.uniform(self.angle_min, self.angle_max)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_CUBIC, 
            borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if labels is not None:
            for i in range(0, len(labels)):
                labels[i] = cv.warpAffine(labels[i], rot_mat, labels[i].shape[1::-1], flags=cv.INTER_NEAREST, borderValue=(0))
        if mask is not None:
            mask = cv.warpAffine(mask, rot_mat, image.shape[1::-1], flags=cv.INTER_NEAREST, borderValue=(0))
        return image, labels, mask


class ConvertFromInts(object):
    def __call__(self, image, labels=None, mask=None):
        image = image.astype(np.float32)
        if labels is not None:
            for i in range(0, len(labels)):
                labels[i] = labels[i].astype(np.float32)
        if mask is not None:
            mask = mask.astype(np.float32)
        return image, labels, mask


class ConvertToInts(object):
    def __call__(self, image, labels=None, mask=None):
        image = image.astype(np.uint8)
        if labels is not None:
            for i in range(0, len(labels)):
                labels[i] = labels[i].astype(np.uint8)
        if mask is not None:
            mask = mask.astype(np.uint8)
        return image, labels, mask


class Normalize(object):
    def __call__(self, image, labels=None, mask=None):
        image = image.astype(np.float32) / 255.0
        # if label is not None:
        #     label = label.astype(np.float32) / 255.0
        # if mask is not None:
        #     mask = mask.astype(np.float32) / 255.0
        return image, labels, mask


class Standardize(object):
    def __init__(self, mean_rgb:list=[0.485, 0.456, 0.406], std_rgb:list=[0.229, 0.224, 0.225]):
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb

    def __call__(self, image_bgr, labels=None, mask=None):
        image_bgr[:,:,0] = (image_bgr[:,:,0] - self.mean_rgb[2]) / self.std_rgb[2]
        image_bgr[:,:,1] = (image_bgr[:,:,1] - self.mean_rgb[1]) / self.std_rgb[1]
        image_bgr[:,:,2] = (image_bgr[:,:,2] - self.mean_rgb[0]) / self.std_rgb[0]
        return image_bgr, labels, mask


class RandomMirror(object):
    def __init__(self, horizont_prob:float=0.5, probabilty:float=0.5):
            self.horizont_prob = np.clip(horizont_prob, 0.0, 1.0)
            self.probabilty = np.clip(probabilty, 0.0, 1.0)

    def __call__(self, image, labels=None, mask=None):
        do_flip = np.random.choice([0, 1], size=1, p=[1-self.probabilty, self.probabilty])
        is_hori  = np.random.choice([0, 1], size=1, p=[1-self.horizont_prob, self.horizont_prob])
        if do_flip:
            if is_hori:
                image = image[:, ::-1]
                if labels is not None:
                    for i in range(0, len(labels)):
                        labels[i] = labels[i][:, ::-1]
                if mask is not None:
                    mask = mask[:,::-1]
            else:
                image = image[::-1]
                if labels is not None:
                    for i in range(0, len(labels)):
                        labels[i] = labels[i][::-1]
                if mask is not None:
                    mask = mask[::-1]
        return image, labels, mask


class RandomGamma(object):
    def __init__(self, lower:float=0.5, upper:float=2.0, probabilty:float=0.5):
        self.lower = lower
        self.upper = upper
        self.probabilty = np.clip(probabilty, 0.0, 1.0)
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, labels=None, mask=None):
        assert image.dtype == np.float32, "image dtype must be float"
        if np.random.choice([0, 1], size=1, p=[1-self.probabilty, self.probabilty]):
            gamma = np.random.uniform(self.lower, self.upper)
            if np.mean(image) > 100:
                image = pow(image / 255., gamma) * 255.
        return image, labels, mask


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, labels=None, mask=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, labels, mask


class RandomHue(object):
    def __init__(self, delta:float=30.0, probabilty:float=0.5):
        self.delta = np.clip(delta, 0.0, 360.0)
        self.probabilty = np.clip(probabilty, 0.0, 1.0)

    def __call__(self, image, labels=None, mask=None):
        if np.random.choice([0, 1], size=1, p=[1-self.probabilty, self.probabilty]):
            cvt = ConvertColor(current="RGB", transform='HSV')
            image, _, _ = cvt(image)
            ru = np.random.uniform(-self.delta, self.delta)
            image[:, :, 0] += ru
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
            cvt = ConvertColor(current="HSV", transform='RGB')
            image, _, _ = cvt(image)
        return image, labels, mask


class Clip(object):
    def __init__(self, mmin:float=0.0, mmax:float=255.0):
        self.min = mmin
        self.max = mmax
        assert self.max >= self.min, "min val must be >= max val"

    def __call__(self, image, labels=None, mask=None):
        image = np.clip(image, self.min, self.max)
        return image, labels, mask


class ToTensor(object):
    def __init__(self, norm_label:bool=True, norm_mask:bool=True):
        self.norm_label = norm_label
        self.norm_mask = norm_mask

    def __call__(self, cvimage, labels=None, mask=None):
        img = torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)

        if labels is not None:
            for i in range(0, len(labels)):
                labels[i] = labels[i].astype(np.float32)
                if self.norm_label:
                    labels[i] = labels[i] / 255.0
            labels = torch.from_numpy(np.stack(labels, axis=0))

        if mask is not None:
            mask = mask.astype(np.float32)
            if self.norm_mask:
                mask = mask / 255.0
            mask = torch.from_numpy(np.expand_dims(mask, 0))

        return img, labels, mask


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))