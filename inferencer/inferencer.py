import numpy as np
import os
import torch
import cv2
import pandas as pd
from model import resnet


class Inferencer(object):

    def __init__(self, img_dir, model_path):
        self.img_dir = img_dir
        self.model = self.load_model(model_path)

    def load_model(self, model_path):

        model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def run(self):
        pass
        # TODO: img Normalize, batch or single,