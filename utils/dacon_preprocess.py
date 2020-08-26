import os
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':

    input_dir = '../data/dacon_cls/'
    label_map_path = '../dataloader/dataset/labelmap/dacon_cls.name'
    input_path = os.path.join(input_dir, 'train.csv')
    LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
                        'L': 11,
                        'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
                        'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
    switch_kv_LETTER_DICT = {v: k for k, v in LETTER_DICT.items()}

    train_set = pd.read_csv(input_path)

    #Digit이 있는 영역을 class별 영상으로 저장해서 하나씩 확인해보자
    imgs = train_set.loc[:, '0':]
    letter = train_set.loc[:, 'letter']
    digit = train_set.loc[:, 'digit']


    for i in range(len(LETTER_DICT)):
        letter_pd = train_set.loc[train_set['letter'] == switch_kv_LETTER_DICT[i], :]
        letter_pd.to_csv('../data/dacon_cls/' + str(switch_kv_LETTER_DICT[i]) + '.csv', sep=',', index=False)

    digit_save_dir = os.path.join('/home/jsk/data/dacon_cls/', 'digit')
    letter_save_dir = os.path.join('/home/jsk/data/dacon_cls/', 'letter')

    for i in range(len(train_set)):
        img = np.array(imgs.loc[i, :]).reshape(28, 28).astype(np.uint8)

        letter_img = cv2.inRange(img, 50, 160)
        digit_img = cv2.inRange(img, 161, 255)

        let = str(letter[i])
        dit = str(digit[i])

        let_save_dir = os.path.join(letter_save_dir, let)
        dit_save_dir = os.path.join(digit_save_dir, dit)

        if not os.path.exists(let_save_dir):
            os.makedirs(let_save_dir)
        if not os.path.exists(dit_save_dir):
            os.makedirs(dit_save_dir)

        cv2.imwrite(os.path.join(let_save_dir, let + '_' + dit + '_' + str(i) + '.jpg'), letter_img)
        cv2.imwrite(os.path.join(dit_save_dir, let + '_' + dit + '_' + str(i) + '.jpg'), digit_img)



