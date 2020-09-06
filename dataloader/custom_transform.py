from torch.jit.annotations import List, Dict, Tuple
import random
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms.functional as F


def transforms_train():
    composed_transform = Compose([
                                  RandomShift(),
                                  RandomRotation(),
                                  Normalize(),
                                  ToTensor()])
    return composed_transform


def transforms_test():
    composed_transform = Compose([
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
        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR)
        sample = {'img': img, 'letter': letter, 'target': target}

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
        img = sample['img']
        h, w = img.shape[:2]
        angle = self.get_params(self.degree)
        (cX, cY) = (w/2, h/2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, scale=1.0)
        dst = cv2.warpAffine(img, M, (w, h))

        sample['img'] = dst

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
        img = sample['img']
        h, w = img.shape[:2]
        shift_x, shift_y = self.get_params(self.translate)

        M = np.array([[1, 0, shift_x], [0, 1, shift_y]])
        dst = cv2.warpAffine(img, M, (w, h))

        sample['img'] = dst

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