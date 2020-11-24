import os
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as td
import torchvision as tv


class ExtendedYaleFace(td.Dataset):
    """Cropped Extended Yale Face Database B."""
    def __init__(self,
                 root="data/CroppedYale",
                 image_shape=[32, 32],
                 flatten=False,
                 normalize=False,
                 train=False,
                 test=False):
        
        corrputed = [
            "yaleB11_P00A+050E-40.pgm",
            "yaleB11_P00A+095E+00.pgm",
            "yaleB11_P00A-050E-40.pgm",
            "yaleB11_P00A-110E+15.pgm",
            "yaleB12_P00A+050E-40.pgm",
            "yaleB12_P00A+095E+00.pgm",
            "yaleB12_P00A-050E-40.pgm",
            "yaleB12_P00A-110E+15.pgm",
            "yaleB12_P00A-110E-20.pgm",
            "yaleB13_P00A+050E-40.pgm",
            "yaleB13_P00A+095E+00.pgm",
            "yaleB13_P00A-050E-40.pgm",
            "yaleB13_P00A-110E+15.pgm",
            "yaleB15_P00A-035E+40.pgm",
            "yaleB16_P00A+095E+00.pgm",
            "yaleB16_P00A-010E+00.pgm",
            "yaleB17_P00A-010E+00.pgm",
            "yaleB18_P00A-010E+00.pgm"
            ]

        assert train or test

        self.root = root
        self.image_shape = image_shape
        self.flatten = flatten
        self.normalize = normalize

        self.image_path = []
        self.labels = []

        for label, subj_folder in enumerate(sorted(os.listdir(self.root))):
            temp_folder = os.path.join(self.root, subj_folder)
            for image in sorted(glob.glob(os.path.join(temp_folder, '*.pgm'))):
                if ('Ambient' in image) or(image in corrputed):
                    continue
                self.image_path.append(image)
                self.labels.append(label)

        assert len(self.image_path) == len(self.labels)

        train_path, test_path, train_labels, test_labels = \
                    train_test_split(self.image_path, self.labels, test_size=0.2, random_state=42)
        if train:
            self.image_path = train_path
            self.labels = train_labels
        elif test:
            self.image_path = test_path
            self.labels = test_labels
        self.len = len(self.image_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = self.image_path[idx]
        label = self.labels[idx]
        img = Image.open(image)
        img = img.resize(self.image_shape)
        img_out = tv.transforms.ToTensor()(img)
        if self.normalize:
            img_out = tv.transforms.Normalize(mean=0.5, std=0.5)(img_out)

        # flatten image
        if self.flatten:
            img_out = img_out.view(1, -1)
        return img_out, label

        
