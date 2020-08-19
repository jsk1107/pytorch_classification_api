import os
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


class TensorboardSummary(object):
    def __init__(self, save_directory):
        self.save_directory = save_directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.save_directory))
        return writer

    def visualize_image(self, writer, image, target, pred, global_step):
        grid_image = make_grid(image, 3, normalize=True)
        writer.add_image(f'Image/{target}', grid_image, global_step)