import os
from torch.utils.data import Dataset
from dataloader.utils import label_map
import cv2

class ClassificationLoader(Dataset):
    def __init__(self, img_path, target, transforms=None):
        self.img_path = img_path
        self.target = target
        self.transforms = transforms

    def __getitem__(self, idx):
        _im = self._load_img(idx)
        _target = self._load_target(idx)

        # cv2.imshow(f'{_target}', _im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        sample = {'image': _im, 'target': _target}

        if self.transforms is not None:
            return self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_path)

    def _load_img(self, idx):
        _im = cv2.imread(self.img_path[idx], cv2.IMREAD_COLOR)
        # _im = cv2.cvtColor(_im, cv2.COLOR_BGR2RGB)
        return _im

    def _load_target(self, idx):
        _target = self.target[idx]
        return _target


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

