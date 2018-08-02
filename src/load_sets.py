import os
import torch  # pylint: disable=import-error
import torch.utils.data  # pylint: disable=import-error
from torch.utils.data import DataLoader  # pylint: disable=import-error
import random

from pathlib import Path
from PIL import Image
import math
import numpy as np
import cv2  # pylint: disable=import-error


def getpnglist(path):
    pngfiles = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(path)
        for name in files
        if name.endswith(".png")
    ]
    return pngfiles


def open_process_image(
    filename, scale_to=[64, 64], is_imageset=False, is_fontset=False
): 
    """Opens an image, returns the preprocessed image (scaled, masked)"""

    if is_imageset:
        img = np.multiply(
            cv2.imread(filename), cv2.imread(filename.replace("Bmp", "Msk"))
        )
        img = np.multiply(img, 1.0 / 255.0)
    else:
        img = cv2.imread(filename)

    processed_img = np.zeros([*scale_to, 3])

    # scaling
    #  img_w, img_h = img.shape[1], img.shape[0]
    #  target_w, target_h = scale_to[1], scale_to[0]
    #  factor = target_w / img_w if img_w/img_h > target_w/target_h else target_h / img_h
    #  img = cv2.resize(img, None, fx=factor, fy=factor)
    img = cv2.resize(img, tuple(scale_to))

    # centering image
    #  x, y = int(target_w/2 - img.shape[1]/2), int(target_h/2 - img.shape[0]/2)
    #  processed_img[y:y+img.shape[0], x:x+img.shape[1]] = img

    # normalising
    processed_img = img.astype(np.float32)
    processed_img /= np.max(processed_img,(0,1),keepdims=True)
    
    # to grayscale

    processed_img = cv2.cvtColor(
        (processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    processed_img = np.expand_dims(processed_img, -1)

    if is_fontset:
        processed_img = (255-processed_img)

    # tests
    processed_img = processed_img.T

    return processed_img


class CharsDataset(torch.utils.data.Dataset):
    PATHS = {"fornts"}

    def __init__(self, name="fontset"):

        self.name = name
        if name == "fontset":
            path = "/data/English/Fnt/"
            self.pngfiles = getpnglist(path)
        elif name == "handset":
            path = "/data/English/Hnd/Img/"
            self.pngfiles = getpnglist(path)
        elif name == "imageset":
            path = "/data/English/Img/GoodImg/Bmp"
            self.pngfiles = getpnglist(path)

    def __len__(self):
        if self.name == "imageset":
            lens = int(len(self.pngfiles) / 2)
        else:
            lens = len(self.pngfiles)
        return lens

    def __getitem__(self, idx):
        if self.name == "fontset":
            path = self.pngfiles[idx]
            image = open_process_image(
                path, is_fontset=True
            )  
            classe = int(path[-13:-10]) - 1
        if self.name == "handset":
            path = self.pngfiles[idx]
            image = open_process_image(path)
            classe = int(path[-11:-8]) - 1
        elif self.name == "imageset":
            path = self.pngfiles[idx]
            image = open_process_image(path, is_imageset=True)
            classe = int(path[-13:-10]) - 1
        elif self.name == "smallfontset":
            path = self.pngfiles[idx]
            image = open_process_image(path)
            classe = int(path[-13:-10]) - 1

        return (image, classe)


def get_datasets():    
    font_dataset = CharsDataset("fontset")
    image_dataset = CharsDataset("imageset")
    hand_dataset = CharsDataset("handset")
    
    fontset = DataLoader(font_dataset, batch_size=512, shuffle=True)
    imageset = DataLoader(image_dataset, batch_size=256, shuffle=True)
    handset = DataLoader(hand_dataset, batch_size=512, shuffle=True)
    
    idx_for_small_fontset = list(range(len(font_dataset)))
    random.shuffle(idx_for_small_fontset)
    idx_for_small_fontset = idx_for_small_fontset[:4000]
    small_fontset = torch.utils.data.dataset.Subset(font_dataset, idx_for_small_fontset)
    
    fat_dataset = torch.utils.data.ConcatDataset([small_fontset, image_dataset])
    
    fatset = DataLoader(fat_dataset, batch_size=256, shuffle=True)
    
    idx_for_small_imageset = list(range(len(image_dataset)))
    random.shuffle(idx_for_small_imageset)
    idx_for_small_imageset = idx_for_small_imageset[:4000]
    small_imageset = torch.utils.data.dataset.Subset(image_dataset, idx_for_small_imageset)
    
    
    idx_for_small_testset = list(range(len(image_dataset)))
    random.shuffle(idx_for_small_testset)
    idx_for_small_testset = idx_for_small_testset[:15]
    small_testset = torch.utils.data.dataset.Subset(image_dataset, idx_for_small_testset)
    
    return {'fontset':fontset,'imageset':imageset,'handset':handset,'small_testset':small_testset,
            'small_imageset':small_imageset,'fat_dataset':fat_dataset,'font_dataset':font_dataset,'image_dataset':image_dataset, 'fatset':fatset,'small_fontset':small_fontset}
