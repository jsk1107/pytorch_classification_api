import os
import torch
from log.saver import Saver
from log.summarise import TensorboardSummary
from log.logger import get_logger
from dataloader import get_dataloader
from model.metric import MetricTracker, accuracy
from tqdm import tqdm
from dataloader.utils import label_map
from efficientnet_pytorch import EfficientNet
from torchvision.models import *
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, label_map, scheduler,
                 matric, cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_map = label_map
        self.cuda = cuda
        self.scheduler = scheduler
        self.matric = matric

    def fit(self, epoch):
        self.train(epoch)
        self.validation(epoch)

    def train(self, epoch):
        self.model.train()
        train_loss = .0

        train_len = self.train_loader.__len__()
        with tqdm(self.train_loader) as tbar:
            for i, sample in enumerate(tbar):
                img = sample['image']
                target = sample['target']
                if self.cuda:
                    img, target = img.cuda(), target.cuda()
                print(target, target.dtype)
                self.optimizer.zero_grad()
                output = self.model(img)
                loss = F.cross_entropy(output, target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description(f'EPOCH : {epoch} | Train loss : {train_loss / (i + 1):.3f}')

        self.scheduler.step()

    def validation(self, epoch):
        self.model.eval()
        self.metric.reset()
        val_loss = .0
        val_len = self.val_loader.__len__()

        with tqdm(self.val_loader) as tbar:
            for i, sample in enumerate(tbar):
                img = sample['image']
                target = sample['target']

                if self.cuda:
                    img, target = img.cuda(), target.cuda()
                print(img, img.dtype)
                print(target, target.dtype)
                with torch.no_grad():
                    output = self.model(img)
                loss = F.cross_entropy(output, target)
                val_loss += loss.item()
                tbar.set_description(f'Validation loss : {val_loss / (i + 1):.3f}')
                print(output)
                self.metric.update(output, target)


        ACC_PER_CATEGORY, mAP, mAR, TOTAL_F1_SCORE, TOTAL_ACC = self.metric.accuracy()

        # save validation_log
        for i in range(len(self.label_map)):
            class_name = list(self.label_map.keys())[i]

            if ACC_PER_CATEGORY.get(class_name) is None:
                continue

        print(f'=============> Accuracy : {TOTAL_ACC}')


if __name__ == '__main__':
    """Unit Test"""
    import argparse
    from parse_config import ParseConfig

    parser = argparse.ArgumentParser('Classification API')
    parser.add_argument('--config-file', '-c', type=str, default='../cfg/config.yaml',
                        help='Config File')
    config = ParseConfig(parser).parse_args()

    print(config)

    train_loader, val_loader, label_map, num_classes = get_dataloader(config)
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.9)
    matric = MetricTracker(label_map)
    trainer = Trainer(model, optimizer, train_loader, val_loader, label_map, scheduler, matric, cuda=False)
    trainer.fit(10)

