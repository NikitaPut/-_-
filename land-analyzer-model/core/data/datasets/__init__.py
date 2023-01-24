from .LandDataset import LandDataset


def build_dataset(root_dir:str, class_labels:list, transforms=None):
    dataset = LandDataset(root_dir, class_labels, transforms)
    return dataset