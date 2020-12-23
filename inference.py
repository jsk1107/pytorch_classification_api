import numpy as np
import os
import torch
from model import network
import cv2
import argparse
from torch.utils.data import Dataset, DataLoader
from dataloader.custom_transform import transforms_val


class InferenceDataset(Dataset):

    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_paths = [os.path.join(self.img_dir, f.name) for f in os.scandir(self.img_dir)]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        dummy = torch.empty(0)
        sample = {'image': img, 'target': dummy}
        if self.transforms is not None:
            sample = self.transforms(sample)
            sample['img_path'] = img_path
            return sample
        sample['origin_img'] = img
        return sample

    def __len__(self):
        return len(self.img_paths)


class Inferencer(object):

    def __init__(self, args):
        self.args = args
        dataset = InferenceDataset(self.args.img_dir, transforms=transforms_val(self.args))
        self.dataloader = DataLoader(dataset, batch_size=3, num_workers=4)
        self.model, self.label_map = self.load_model(self.args.model_path)

    def load_model(self, model_path):

        checkpoint = torch.load(model_path)
        model_name = checkpoint['network']
        label_map = checkpoint['label_map']
        model = network(model_name, pretrained=False, num_classes=len(label_map))
        model.load_state_dict(checkpoint['state_dict'])

        return model, label_map

    def inference(self):

        self.model.eval()
        for i, sample in enumerate(self.dataloader):

            img = sample['image']
            if torch.cuda.is_available():
                img = img.cuda()

            with torch.no_grad():
                output = self.model(img)

            output = output.cpu().numpy()
            category_idx = np.argmax(output, axis=1)

            # batch로 들어온 이미지들의 추론된 label명 담기
            label = []
            for idx in category_idx:
                label.append(self.label_map[idx])
            img_path = sample['img_path']

            # save img. img_path, label의 인덱스 순서는 매칭되어있는 상태임.
            self.save_img(img_path, label)

    def save_img(self, img_paths, label):

        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir, exist_ok=True)

        for i, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            label_dir = os.path.join(self.args.save_dir, label[i])
            if not os.path.isdir(label_dir):
                os.makedirs(label_dir, exist_ok=True)

            cnt = len([f for f in os.scandir(label_dir)])
            img_name = os.path.join(label_dir, str(cnt+1).zfill(4) + '.jpg')
            cv2.imwrite(img_name, img)
            print(f'Save!! ===========> {img_name}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Inference Classification')
    parser.add_argument('--img-dir', '-i', type=str, default='./demo/', help='img dir for inference')
    parser.add_argument('--save-dir', type=str, default='./output', help='Path of the file to be saved')
    parser.add_argument('--model-path', '-m', type=str, default='./run/test/efficientnet-b0/model_best.pth.tar', help='img dir for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch_size to be used for inference. If you want to infer a single image, enter 1.')
    parser.add_argument('--resize', type=int, default=[224, 224], help='Image size [H, W] to be adjusted. It is recommended to match the input image size used in training.')

    args = parser.parse_args()

    inferencer = Inferencer(args)
    inferencer.inference()
