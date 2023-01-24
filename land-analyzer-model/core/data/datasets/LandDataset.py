import cv2 as cv
import numpy as np
from glob import glob
from torch.utils.data import Dataset


class LandDataset(Dataset):
    def __init__(self, root_dir:str, class_labels:list, transforms=None):
        self.root_dir = root_dir
        self.imgs = sorted(glob(root_dir + "/*/img/*"))
        self.labels = {}
        for cl in class_labels:
            self.labels[cl] = sorted(glob(root_dir + "/*/{0}/*".format(cl)))
            assert len(self.imgs) == len(self.labels[cl])

        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Read image
        img_path = self.imgs[idx]
        image = cv.imread(img_path)

        # Read labels
        item_labels = []
        for key in self.labels.keys():
            label_path = self.labels[key][idx]
            label = cv.imread(label_path)
            label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)
            item_labels.append(label)

        # Create ROI mask
        mask = np.full_like(image, fill_value=255)

        # Prepare data
        if self.transforms:
            image, item_labels, mask = self.transforms(image, item_labels, mask)

        return image, item_labels, mask

    def visualize(self, tick_ms=25):
        for i in range(0, self.__len__()):
            image, labels, mask = self.__getitem__(i)
            cv.imshow('Image', image.astype(np.uint8))
            cv.imshow('Mask', mask.astype(np.uint8))
            for i, label in enumerate(labels):
                cv.imshow('Label_{0}'.format(i), label.astype(np.uint8))
            cv.waitKey(tick_ms)