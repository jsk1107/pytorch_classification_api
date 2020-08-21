import os
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':

    input_dir = '../data/dacon_cls/'

    input_path = os.path.join(input_dir, 'train.csv')

    train_set = pd.read_csv(input_path)

    #Digit이 있는 영역을 class별 영상으로 저장해서 하나씩 확인해보자
    imgs = train_set.loc[:, '0':]
    letter = train_set.loc[:, 'letter']
    digit = train_set.loc[:, 'digit']

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
        #
        # print('letter : ', letter[i])
        # print('digit : ', digit[i])
        #
        # cv2.imshow('origin_img', img)
        # cv2.imshow('letter_img', letter_img)
        # cv2.imshow('digit_img', digit_img)
        # cv2.waitKey(0)



