import os
import argparse
from parse_config import ParseConfig
import torch
from dataloader import get_dataloader
from model.metric import MetricTracker, accuracy
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def run():
    parser = argparse.ArgumentParser('Classification API')
    parser.add_argument('--config-file', '-c', type=str, default='./cfg/config.yaml',
                        help='Config File')
    config = ParseConfig(parser).parse_args()

    # Todo: 기본제공되는 ImageFolder, transform 이용해보기
    print(config.root_dir)
    torch.manual_seed(1)

    train_dir = os.path.join(config.root_dir, 'training_set/training_set')
    test_dir = os.path.join(config.root_dir, 'test_set/test_set')

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])
    # train_loader, val_loader, label_map, num_classes = get_dataloader(config)

    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)
    label_map = train_dataset.class_to_idx
    num_classes = len(label_map)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    # model = demirnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()
    metric = MetricTracker(label_map)
    schduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.milestones, config.gamma)
    for epoch in range(0, 200):
      # optimizer.zero_grad()
      model.train()
      train_loss = .0
      with tqdm(train_loader) as tbar:
        for i, sample in enumerate(tbar):
          img = sample[0]
          target = sample[1]
          if config.cuda:
            img, target = img.cuda(), target.cuda()
            model = model.cuda()
          optimizer.zero_grad()
          output = model(img)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()
          tbar.set_description(f'EPOCH : {epoch} | Train loss : {train_loss / (i + 1):.3f}')
      schduler.step()
      model.eval()
      # for m in model.modules():
      #   if isinstance(m, nn.BatchNorm2d):
      #     m.track_runing_stats=False
      metric.reset()
      val_loss = .0
      val_len = val_loader.__len__()

      with tqdm(val_loader) as tbar:
          for i, sample in enumerate(tbar):
              img = sample[0]
              target = sample[1]

              if config.cuda:
                  img, target = img.cuda(), target.cuda()
              with torch.no_grad():
                  output = model(img)
              loss = criterion(output, target)
              val_loss += loss.item()
              tbar.set_description(f'Validation loss : {val_loss / (i + 1):.3f}')
              metric.update(output, target)
      ACC_PER_CATEGORY, mAP, mAR, TOTAL_F1_SCORE, TOTAL_ACC = metric.accuracy()
      print(TOTAL_ACC)

if __name__ == '__main__':
    run()