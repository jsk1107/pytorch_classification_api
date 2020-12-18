import numpy as np
from dataloader.dataset import dataset
from dataloader.custom_transform import transforms_train, transforms_test
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_dataloader(config):

    train_dataset = dataset.ClassificationLoader(config.root_dir,
                                                 config.label_map_path,
                                                 split='train',
                                                 transform=transforms_train(config))

    test_dataset = dataset.ClassificationLoader(config.root_dir,
                                                config.label_map_path,
                                                split='test',
                                                transform=transforms_test(config))

    label_map = train_dataset.classes

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              pin_memory=config.pin_memory,
                              num_workers=config.num_workers)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                              pin_memory=config.pin_memory,
                              num_workers=config.num_workers)


    return train_loader, test_loader, label_map