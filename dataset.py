from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import decode_image
from synthetic import convert

import cv2
import numpy as np

import torch
import glob, os, json

synth_transforms = v2.Compose([
    v2.Resize((256, 256)),
    v2.RandomCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5])
])

synth_dir = '/home/john/Downloads/synthetic_v2/'

def parse_tuples(obj):
    it = obj.items() if type(obj) is dict else enumerate(obj)
    for key, value in it:
        if type(value) is str and value[0] == '(' and value[-1] == ')': # float tuple as string, (0.00, 0.00, 0.00)
            items = value[1:-1].split(', ')
            obj[key] = tuple(float(item) for item in items)
        elif type(value) in (list, dict): # dict/lists, recursion
            obj[key] = parse_tuples(value)
        elif type(value) == str: # floats as strings (usually)
            try:
                obj[key] = float(value)
            except ValueError:
                pass
    return obj

class SyntheticDS(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.img_dir = sorted(glob.glob(os.path.join(synth_dir, 'images', '*.jpg')), key=lambda x: int(os.path.basename(x).removesuffix('.jpg')))
        self.lbl_dir = sorted(glob.glob(os.path.join(synth_dir, 'labels', '*.json')), key=lambda x: int(os.path.basename(x).removesuffix('.json')))
        self.transforms = transforms
    def __len__(self):
        return min(len(self.img_dir), len(self.lbl_dir))
    def __getitem__(self, idx):
        img = decode_image(self.img_dir[idx])
        with open(self.lbl_dir[idx], 'r') as lbl_file:
            lbl = json.load(lbl_file, object_hook=parse_tuples)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, convert(lbl)