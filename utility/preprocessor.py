""" Preprocessor of frames """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.img_height = self.config['img_height']
        self.img_width = self.config['img_width']

    def normalize_frames_train(self, frame): # Windows does not support pickle of lambda funcs
        normalize_transforms = [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        transpose_function = transforms.Compose(normalize_transforms)
        return transpose_function(frame)

    def normalize_frames_test(self, frame):  # Windows does not support pickle of lambda funcs
        normalize_transforms = [transforms.ToPILImage(), transforms.ToTensor()]
        transpose_function = transforms.Compose(normalize_transforms)
        return transpose_function(frame)

    def resize_frame(self, frame):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor()
        ])
        return transform(frame)

    def downsample_normalize_frames(self, frames):
        # 0=batch size, 3=img channels, 1 and 2 = img dims, / 255 normalize
        transform = transforms.Lambda(lambda img: np.transpose(img, (0, 3, 1, 2)) / 255)
        normalized_frames = transform(frames)
        downsized_frames = F.interpolate(torch.Tensor(normalized_frames), size=(self.img_height, self.img_width), mode='bicubic', align_corners=True)
        return downsized_frames
