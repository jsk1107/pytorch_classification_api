import os
import torch
from dataloader import get_dataloader
from model import resnet
from log.saver import Saver
from log.summarise import TensorboardSummary
from log.logger import get_logger
from dataloader import get_dataloader
from model.metric import MetricTracker, accuracy
from tqdm import tqdm
from dataloader.utils import label_map


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
        self.train_loader, self.val_loader, self.label_map = get_dataloader(self.config)
        self.label_map = label_map(self.config.label_map_path)
        # 임시?? Letter Map
        self.LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                   'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
                   'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
        # Define Network(Resnet50)
        self.model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10)

        # Define Optim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # Define Scheduler
        self.schduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config.gamma)

        # TODO: Loss
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
        if self.config.resume:
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
                img = sample['img']
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
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + epoch * train_len)
                self.tensorboardsummary.visualize_image(self.writer, img[0], target[0], output[0], i)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

        self.model.eval()
        self.metric.reset()
        val_loss = .0
        val_len = self.train_loader.__len__()

        with tqdm(self.train_loader) as tbar:
            for i, sample in enumerate(tbar):
                img = sample['img']
                target = sample['target']

                if self.config.cuda:
                    img, target = img.cuda(), target.cuda()

                with torch.no_grad():
                    print(img.size())
                    output = self.model(img)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                tbar.set_description(f'Validation loss : {val_loss / (i + 1):.3f}')
                self.writer.add_scalar('validation/val_loss_iter', loss.item(), i + epoch * val_len)

                self.metric.update(target, output)
            self.logger.info(f'Cunfusion Metric : Row is True, Col is Pred. \n {self.metric.result()}')

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
        # if TOTAL_ACC > self.best_pred:
        is_best = False
        self.best_pred = TOTAL_ACC
        state = self.model.module
        # state = {'best_pred': TOTAL_ACC,
        #          'epoch': epoch + 1,
        #          'state_dict': self.model.state_dict(),
        #          'module': self.model.modules(),
        #          'optimizer': self.optimizer.state_dict()}
        self.saver.save_checkpoint(state, is_best)


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