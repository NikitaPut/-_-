import logging
from .datasets import build_dataset
from .transforms import build_transforms
from torch.utils.data import Dataset, ConcatDataset, RandomSampler, SequentialSampler, BatchSampler, DataLoader


def create_loader(dataset:Dataset,
                  shuffle:bool,
                  batch_size:int,
                  num_workers:int = 1,
                  pin_memory:bool = True):
    
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, pin_memory=pin_memory)
    return data_loader


def make_data_loader(cfg, is_train:bool = True) -> DataLoader:
    logger = logging.getLogger('CORE')

    transforms = build_transforms(cfg.INPUT.IMAGE_SIZE, is_train=is_train, to_tensor=True)

    # Create datasets
    datasets = []

    if is_train:
        cfg_dataset_node = cfg.DATASETS.TRAIN
    else:
        cfg_dataset_node = cfg.DATASETS.TEST

    for root_dir in cfg_dataset_node.ROOT_DIR:
        dataset = build_dataset(root_dir, cfg.MODEL.HEAD.CLASS_LABELS, transforms)
        logger.info("Loaded dataset from '{0}'. Size: {1}".format(root_dir, dataset.__len__()))
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)

    # Create data loader
    batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
    shuffle = is_train

    data_loader = create_loader(dataset, shuffle, batch_size, cfg.DATA_LOADER.NUM_WORKERS, cfg.DATA_LOADER.PIN_MEMORY)

    return data_loader