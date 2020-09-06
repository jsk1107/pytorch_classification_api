import numpy as np
import os
import torch
import cv2
import pandas as pd

### 파일 불러오기

test_path = './data/dacon_cls/test.csv'
submission = './data/dacon_cls/submission.csv'
model_path = './run/Dacon_cls/resnet-50/'

test_dataset = pd.read_csv(test_path)
submission_dataset = pd.read_csv(submission)

model = torch.load(os.path.join(model_path, '20200830_0', 'checkpoint.pth.tar'))
model.eval()

for j in range(len(test_dataset)):
    id = test_dataset.loc[j, 'id']
    img = test_dataset.loc[j, '0':]
    img = np.array(img).reshape(28, 28).astype(np.uint8) / 255.0
    digit_img = torch.from_numpy(img).type(torch.FloatTensor)
    digit_img = digit_img.unsqueeze(0)
    digit_img = digit_img.unsqueeze(0)
    output = model(digit_img.cuda90)
    pred_digit = torch.argmax(output, dim=1).cpu().data.numpy()
    submission_dataset.loc[submission_dataset['id'] == id, 'digit'] = pred_digit
submission_dataset.to_csv(submission, sep=',', index=False)