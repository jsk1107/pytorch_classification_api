import numpy as np
from dataloader.dataset import dataset
from dataloader.custom_transform import transforms_train, transforms_test
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_dataloader(config):

    if config.project_name == 'cifar10':
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

    elif config.project_name == 'Dacon_cls':
        train_dataset = dataset.DaconDataloader(config.root_dir, config.label_map_path, split='train', transforms=transforms_train())
        label_map = train_dataset.classes

        train_cnt = train_dataset.__len__()
        indices = list(range(train_cnt))

        np.random.seed(216)
        np.random.shuffle(indices)
        split = .7
        train_indices = indices[:int(train_cnt * split)]
        val_indices = indices[int(train_cnt * split):]
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  pin_memory=config.pin_memory,
                                  num_workers=config.num_workers,
                                  sampler=SubsetRandomSampler(train_indices))

        val_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  pin_memory=config.pin_memory,
                                  num_workers=config.num_workers,
                                  sampler=SubsetRandomSampler(val_indices))

        return train_loader, val_loader, label_map