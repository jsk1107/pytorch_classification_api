from torch.jit.annotations import List, Dict, Tuple
import numpy as np
import torch
from PIL import Image
import cv2
import random


def transforms_train(config):
    composed_transform = Compose([Normalize(),
                                  RandomRotation(),
                                  Resize(config.resize),
                                  ToTensor()])
    return composed_transform


def transforms_test(config):
    composed_transform = Compose([Normalize(),
                                  Resize(config.resize),
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
    def __init__(self, mean=(0.485, 0456., 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        target = sample['target']
        img = np.array(img).astype(np.float32)
        target = np.array(target).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std

        sample = {'image': img, 'target': target}

        return sample


class Resize(object):
    def __init__(self, size: List[int]):
        self.width = size[0]
        self.height = size[1]

    def __call__(self, sample):
        img = sample['image']
        target = sample['target']

        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=Image.BILINEAR)

        sample = {'image': img, 'target': target}

        return sample


class RandomRotation(object):
    def __init__(self, degree=30):
        if degree < 0:
            ValueError('It must be positive')
        self.degree = (-degree, degree)

    @staticmethod
    def get_params(degree):
        angle = random.uniform(degree[0], degree[1])

        return angle

    def __call__(self, sample):
        img = sample['image']
        h, w = img.shape[:2]
        angle = self.get_params(self.degree)
        (cX, cY) = (w/2, h/2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, scale=1.0)
        dst = cv2.warpAffine(img, M, (w, h))

        sample['image'] = dst

        return sample


class RandomShift(object):
    def __init__(self, translate=5):
        if translate < 0:
            ValueError('It must be positive')
        self.translate = (-translate, translate)

    @staticmethod
    def get_params(translate):
        shift_x = random.uniform(translate[0], translate[1])
        shift_y = random.uniform(translate[0], translate[1])

        return shift_x, shift_y

    def __call__(self, sample):
        img = sample['image']
        h, w = img.shape[:2]
        shift_x, shift_y = self.get_params(self.translate)

        M = np.array([[1, 0, shift_x], [0, 1, shift_y]])
        dst = cv2.warpAffine(img, M, (w, h))

        sample['image'] = dst

        return sample

class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        target = sample['target']

        # Input Image shape : (H, W, C)
        # Change Image shape : (C, H, W)
        img = np.array(img).astype(np.float32).transpose((2,0,1))
        target = np.array(target).astype(np.int64)

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target)

        sample = {'image': img, 'target': target}

        return sample