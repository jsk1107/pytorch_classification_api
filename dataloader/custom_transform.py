from torch.jit.annotations import List, Dict, Tuple
import numpy as np
import torch
from PIL import Image
import cv2


def transforms_train(config):
    composed_transform = Compose([Resize(config.resize),
                                  ToTensor()])
    return composed_transform


def transforms_test(config):
    composed_transform = Compose([Resize(config.resize),
                                  Normalize(),
                                  ToTensor()])
    return composed_transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Normalize(object):
    def __init__(self, mean=0.485, std=0.229):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['img']
        target = sample['target']
        letter = sample['letter']
        img = np.array(img).astype(np.float32)
        target = np.array(target).astype(np.int32)
        letter = np.array(letter).astype(np.int32)

        img /= 255.0
        # img -= self.mean
        # img /= self.std

        sample = {'img': img, 'letter': letter, 'target': target}

        return sample


class Resize(object):
    def __init__(self, size):
        self.width = int(size[0])
        self.height = int(size[1])

    def __call__(self, sample):
        img = sample['img']
        target = sample['target']
        letter = sample['letter']
        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
        sample = {'img': img, 'letter': letter, 'target': target}

        return sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['img']
        target = sample['target']
        letter = sample['letter']

        # Input Image shape : (H, W, C)
        # Change Image shape : (C, H, W)

        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        img = np.array(img).astype(np.float32)
        target = np.array(target).astype(np.int32)
        letter = np.array(letter).astype(np.int32)

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).type(torch.LongTensor)
        letter = torch.from_numpy(letter).type(torch.LongTensor)

        sample = {'img': img, 'letter': letter, 'target': target}

        return sample