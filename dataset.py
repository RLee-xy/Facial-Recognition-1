import os
import numpy as np
import glob
from PIL import Image

import torch
import torch.utils.data as td
import torchvision as tv


class ExtendedYaleFace(td.Dataset):
    """Cropped Extended Yale Face Database B."""
    def __init__(self,
                 root="data/CroppedYale",
                 image_shape=[32, 32],
                 form='numpy'):

        assert form in ['numpy', 'torch']
        self.form = form
        
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

        self.root = root
        self.image_shape = image_shape

        self.image_path = []
        self.labels = []
        self.len = 0

        for label, subj_folder in enumerate(sorted(os.listdir(self.root))):
            temp_folder = os.path.join(self.root, subj_folder)
            for image in sorted(glob.glob(os.path.join(temp_folder, '*.pgm'))):
                if ('Ambient' in image) or(image in corrputed):
                    continue
                self.image_path.append(image)
                self.labels.append(label)
                self.len += 1

        assert len(self.image_path) == len(self.labels) == self.len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = self.image_path[idx]
        label = self.labels[idx]
        img = Image.open(image)
        img = img.resize(self.image_shape)
        if self.form == "numpy":
            img_out = np.array(img)
            img_out = np.expand_dims(img_out, axis=0)
        elif self.form == "torch":
            img_out = tv.transforms.ToTensor(img)
            img_out = img_out.unsqueeze(0)
        else:
            raise ValueError("output has to be numpy.ndarray or tensor.Tensor")
        return img_out, label

        
