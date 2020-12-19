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

class Trainer(object):
    def __init__(self, config):
        self.config = config

        # Define Saver
        self.saver = Saver(self.config)

        # Define Tensorboard
        if self.config.tensorboard:
            self.tensorboardsummary = TensorboardSummary(self.saver.expriment_dir)
            self.writer = self.tensorboardsummary.create_summary()

        # Define Logger
        self.logger = get_logger(self.config, self.saver.expriment_dir)

        # Define DataLoader
        self.train_loader, self.val_loader, self.label_map, self.num_classes = get_dataloader(self.config)

        # Define Network
        network_name = self.config.model.split()[0]

        if network_name != 'efficientnet':
            print(f'==> Load pretrained weight.')
            self.model = self.network(self.config.model, pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
            print(f'==> final layer channel is modified to {self.num_classes}')
        else:
            self.model = self.network(self.config.model, self.num_classes)

        # Define Optim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # Define Scheduler
        self.schduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.config.milestones, self.config.gamma)

        # Define Loss
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define Metric
        self.metric = MetricTracker(self.label_map)

        # Define CUDA
        if self.config.cuda:
            if not torch.cuda.is_available():
                raise ValueError('==> Cuda is not available. Process is used to CPU')
            print('==> Cuda is available')
            self.model = torch.nn.DataParallel(self.model, device_ids=[self.config.gpu_ids])
        else:
            print('==> Cuda is not available. CPU Only')
            self.model = torch.nn.DataParallel(self.model)

        # Define resume
        self.best_pred = .0
        if self.config.resume is not None:
            if not os.path.isfile(self.config.resume):
                raise FileNotFoundError('=> no checkpoint found at {}'.format(self.config.resume))
            checkpoint = torch.load(self.config.resume)
            self.config.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(f'loaded checkpoint {self.config.resume} epoch {checkpoint["epoch"]}')

    def train(self, epoch):

        self.model.train()
        train_loss = .0

        train_len = self.train_loader.__len__()
        with tqdm(self.train_loader) as tbar:
            for i, sample in enumerate(tbar):
                img = sample['image']
                target = sample['target']

                if self.config.cuda:
                    img, target = img.cuda(), target.cuda()

                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description(f'EPOCH : {epoch} | Train loss : {train_loss / (i + 1):.3f}')

                if self.config.tensorboard:
                    self.writer.add_scalar('train/total_loss_iter', loss.item(), i + epoch * train_len)
                    self.tensorboardsummary.visualize_image(self.writer, img[0], target[0], i)

        if self.config.tensorboard:
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

        self.schduler.step()

    def validation(self, epoch):
        self.model.eval()
        self.metric.reset()
        val_loss = .0
        val_len = self.val_loader.__len__()

        with tqdm(self.val_loader) as tbar:
            for i, sample in enumerate(tbar):
                img = sample['image']
                target = sample['target']

                if self.config.cuda:
                    img, target = img.cuda(), target.cuda()

                with torch.no_grad():
                    output = self.model(img)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                tbar.set_description(f'Validation loss : {val_loss / (i + 1):.3f}')
                self.metric.update(output, target)

                if self.config.tensorboard:
                    self.writer.add_scalar('validation/val_loss_iter', loss.item(), i + epoch * val_len)

            self.logger.info(f'Epoch: {epoch} || Cunfusion Metric: Row is True, Col is Pred. \n {self.metric.result()}')

        ACC_PER_CATEGORY, mAP, mAR, TOTAL_F1_SCORE, TOTAL_ACC = self.metric.accuracy()

        # save validation_log
        for i in range(len(self.label_map)):
            class_name = list(self.label_map.keys())[i]

            if ACC_PER_CATEGORY.get(class_name) is None:
                continue

            self.logger.info(f'Epoch : {epoch} | '
                             f'AR, AP, ACC of {class_name} : '
                             f'{100 * ACC_PER_CATEGORY[class_name]["AR"]:.3f} % '
                             f'{100 * ACC_PER_CATEGORY[class_name]["AP"]:.3f} % '
                             f'{100 * ACC_PER_CATEGORY[class_name]["ACC"]:.3f} % ')
        self.logger.info(f'TOTAL_ACC : {100 * TOTAL_ACC:.3f} %')
        print(f'=============> Accuracy : {TOTAL_ACC}')
        # save checkpoint of best_model
        if TOTAL_ACC > self.best_pred:
            is_best = True
            self.best_pred = TOTAL_ACC
            state = {'best_pred': TOTAL_ACC,
                     'epoch': epoch + 1,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
            self.saver.save_checkpoint(state, is_best)


    def network(self, model, pretrained=True, num_classes=None):

        if model == 'resnet-18':
            model = resnet18(pretrained=pretrained, progress=True)
        elif model == 'resnet-34':
            model = resnet34(pretrained=pretrained, progress=True)
        elif model == 'resnet-50':
            model = resnet50(pretrained=pretrained, progress=True)
        elif model == 'resnet-101':
            model = resnet101(pretrained=pretrained, progress=True)
        elif model == 'resnet-152':
            model = resnet152(pretrained=pretrained, progress=True)
        elif model == 'resnext-50':
            model = resnext50_32x4d(pretrained=pretrained, progress=True)
        elif model == 'resnext-101':
            model = resnext101_32x8d(pretrained=pretrained, progress=True)
        elif model == 'vgg-19':
            model = vgg19(pretrained=pretrained, progress=True)
        elif model == 'inception-v3':
            model = inception_v3(pretrained=pretrained, progress=True)
        elif model == 'mobilenet-v2':
            model = mobilenet_v2(pretrained=pretrained, progress=True)
        elif model == 'mobilenet-v2':
            model = mobilenet_v2(pretrained=pretrained, progress=True)
        elif model == 'efficientnet-b0':
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        elif model == 'efficientnet-b1':
            model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
        elif model == 'efficientnet-b2':
            model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
        elif model == 'efficientnet-b3':
            model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
        elif model == 'efficientnet-b4':
            model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        elif model == 'efficientnet-b5':
            model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
        elif model == 'efficientnet-b6':
            model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
        elif model == 'efficientnet-b7':
            model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
        else:
            raise ImportError(f'=========> It is not found the {model}')

        return model

if __name__ == '__main__':
    """Unit Test"""
    import argparse

    parser = argparse.ArgumentParser('trainer test')

    # Project Name
    parser.add_argument('--dataset', type=str, default='sealing', help='Write your project name')

    # Data Dir
    parser.add_argument('--root_dir', type=str,
                        default='/home/jsk/data/cifar/',
                        help='Image Dir')
    parser.add_argument('--checkname', type=str, default='EfficientNet',
                        help='used model name')

    # Used Model
    parser.add_argument('--project_name', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='efficientnet-b1',
                        help='model type that can be used is'
                             'efficientnet-b0 ~ 6')
    parser.add_argument('--tensorboard', default=True)

    # Data Setting
    parser.add_argument('--classes', type=int, default=10, help='Image size')
    # parser.add_argument('--resize', type=int, default=(256, 1250), help='Image size')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='validation set ratio')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--num-workers', type=int, default=1, help='# cpu worker')
    parser.add_argument('--pin-memory', action='store_true', default=True, help='on/off pin')
    parser.add_argument('--num-classes', type=int, default=7, help='the number of classes')
    parser.add_argument('--epoch', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch')

    # Optimizer, lr, lr_scheduler Setting
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--step-size', type=int, default=100, help='Step over default unit(not epoch. iteration)')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='weight-decay')

    # Env Setting
    parser.add_argument('--cuda', action='store_true', default=False, help='cuda device')
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--fine-tune', default=False, help='finetuning on a different dataset')
    parser.add_argument('--no-val', default=False, help='save checkpoint every epoch')

    args = parser.parse_args()

    print(args)
    trainer = Trainer(args)

    for epoch in range(trainer.config.start_epoch, trainer.config.epoch):
        print('epoch : {}'.format(epoch))
        trainer.train(epoch=epoch)
        break
        # trainer.validation(epoch=epoch)

    trainer.writer.close()