import random
import os
import numpy as np
from glob import glob
from dataloader.dataset import dataset
from dataloader.custom_transform import transforms_train, transforms_val
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

def get_dataloader(config):

    print('==> Create label_map & path')
    classes = os.listdir(config.root_dir)
    label_map = {idx: label for idx, label in enumerate(classes)}
    num_classes = len(classes)

    train_path = []
    train_target = np.empty(0)
    val_path = []
    val_target = np.empty(0)

    for i in range(len(label_map)):
        img_data = glob(os.path.join(config.root_dir, label_map[i]) + '/*')
        random.shuffle(img_data)

        # 학습, 검증 이미지 경로 append하기
        train_path += img_data[round(len(img_data) * config.val_size):]
        val_path += img_data[:round(len(img_data) * config.val_size)]

        # 학습, 검증 이미지 갯수만큼 target label 만들기
        img_cnt = len(img_data[round(len(img_data) * config.val_size):])
        train_target = np.append(train_target, np.repeat(i, img_cnt))
        img_cnt = len(img_data[:round(len(img_data) * config.val_size)])
        val_target = np.append(val_target, np.repeat(i, img_cnt))

    # comb = list(zip(train_path, train_target))
    # random.shuffle(comb)
    # train_path, train_target = zip(*comb)
    print('Done!')

    train_dataset = dataset.ClassificationLoader(train_path,
                                                 train_target,
                                                 transforms=transforms_train(config))

    val_dataset = dataset.ClassificationLoader(val_path,
                                               val_target,
                                               transforms=transforms_val(config))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=config.pin_memory,
                              num_workers=config.num_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config.batch_size,
                            pin_memory=config.pin_memory,
                            num_workers=config.num_workers)

    return train_loader, val_loader, label_map, num_classes
