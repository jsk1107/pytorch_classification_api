import numpy as np
import os
import torch
import cv2
import pandas as pd

### 파일 불러오기

test_path = './data/dacon_cls/test.csv'
submission = './data/datcon_cls/submission.csv'
model_path = './run/Dacon_cls/resnet-50/'

test_dataset = pd.read_csv(test_path)
submission_dataset = pd.read_csv(submission)

LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
                    'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
switch_kv_LETTER_DICT = {v: k for k, v in LETTER_DICT.items()}

for k in range(len(LETTER_DICT)):
    letter_pd = test_dataset.loc[test_dataset['letter'] == switch_kv_LETTER_DICT[k], :]
    letter_pd = letter_pd.reset_index(inplace=False)
    model = torch.load('모델경로')
    for j in range(len(letter_pd)):
        id = letter_pd.loc[j, 'id']
        img = letter_pd.loc[j, '0':]
        img = np.array(img).reshape(28, 28).astype(np.uint8)
        digit_img = cv2.inRange(img, 161, 255)
        output = model(digit_img)
        pred_digit = torch.argmax(output, dim=1)

        submission_dataset[submission_dataset.loc['id'] == id, 'digit'] = pred_digit


