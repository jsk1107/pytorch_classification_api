import os
from torch.utils.data import Dataset
from dataloader.utils import label_map
import cv2
from PIL import Image
import pandas as pd
import numpy as np


class ClassificationLoader(Dataset):
    def __init__(self, root_dir, label_map_path, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data_list = self._getDataPath()
        self.classes = label_map(label_map_path)

    def __getitem__(self, idx):
        _im = self._load_img(idx)
        _target = self._load_target(idx)
        sample = {'image': _im, 'target': _target}
        if self.split == 'train':
            return self.transform(sample)
        elif self.split == 'test':

            return self.transform(sample)
        else:
            raise FileNotFoundError('Folder name is "train" or "test" only')

    def __len__(self):
        return len(self.data_list)

    def _load_img(self, idx):
        _im = cv2.imread(self.data_list[idx], cv2.IMREAD_COLOR)
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2RGB)
        return _im

    def _load_target(self, idx):
        _class = os.path.splitext(os.path.basename(self.data_list[idx]))[0].split('_')[1]
        _target = self.classes[_class]
        return _target

    def _getDataPath(self):
        data_dir = os.path.join(self.root_dir, self.split)
        data_list = [os.path.join(data_dir, f.name) for f in os.scandir(data_dir)]
        return data_list


class DaconDataloader(Dataset):

    def __init__(self, root_dir, label_map_path, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        self.letter_name = [f.name for f in os.scandir(self.root_dir)]
        self.csv_dataset = pd.read_csv(os.path.join(self.root_dir, split + '.csv'))
        self.LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                   'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
                   'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
        self.kv_swich_LETTER_DICT = {y: x for x, y in self.LETTER_DICT.items()}
        self.classes = label_map(label_map_path)
        # self.letter_pd = self.group_letter()

    def __getitem__(self, idx):

        sample = self.load_data(idx)

        # 데이터 잘 불러와졌는지 체크해보기
        # img = sample['img']
        # img = np.expand_dims(img, 2).astype(np.uint8)
        # print(img.shape)
        # targer = sample['target']
        # letter = sample['letter']
        #
        # print(f'idx : {idx}, digit : {targer} | letter : {letter}')
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.csv_dataset)

    def load_data(self, idx):

        letter = self.csv_dataset.loc[idx, 'letter']
        fc_img = self.csv_dataset.loc[idx, '0':]
        img = np.array(fc_img).reshape(28, 28).astype(np.uint8)

        letter_value = self.LETTER_DICT[letter]
        if self.split == 'train':
            target = self.csv_dataset.loc[idx, 'digit']
            return {'img': img, 'letter': letter_value, 'target': target}
        return {'img': img, 'letter': letter_value}

    # def group_letter(self):
    #     letter_pd = {}
    #     for i in range(len(self.LETTER_DICT)):
    #         letter = self.kv_swich_LETTER_DICT[i]
    #         group = self.csv_dataset.loc[self.csv_dataset['letter'] == letter, :]
    #         letter_pd[letter] = group
    #     return letter_pd


if __name__ == '__main__':
    """Unit Test"""
    from dataloader import get_dataloader
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser('Unit Test')
    parser.add_argument('--project-name', type=str, default='Dacon_cls')
    parser.add_argument('--root_dir', type=str, default='../../data/dacon_cls/',
                        help='root_dir')
    parser.add_argument('--label_map_path', type=str, default='../../dataloader/dataset/labelmap/cifar10.name',
                        help='label_map_path')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch_size')
    parser.add_argument('--resize', type=int, default=[28, 28],
                        help='Image Resize')
    parser.add_argument('--pin-memory', default=True)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    print(args)

    train_loader, test_loader, _ = get_dataloader(args)
    for i, sample in enumerate(train_loader):
        img = sample['img']
        target = sample['target']
        tmp_img = img[0].numpy()
        tmp_img = tmp_img.transpose(1, 2, 0).astype(np.uint8)
        cv2.imshow('img', tmp_img)
        cv2.waitKey(0)
        break;

