import torch
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
import os
import imageio
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np

imageio.plugins.freeimage.download()
imageio.core.util.appdata_dir("imageio")
def print_info(x):
    print('=-' * 20)
    print('shape: ', x.shape)
    print(f'range: [{x.min()}, {x.max()}]')
    print('dtype:', x.dtype)
    print('=-' * 20)

if __name__ == "__main__":
    path = 'dataset/2.tif'
    image = imageio.v2.imread(path)
    print(type(image))
    print(image.dtype)

    totensor = ToTensorV2()
    tofloat = A.ToFloat(max_value=65535)

    image_f = tofloat(image=image)['image']
    image_t = totensor(image=image_f)['image']
    print_info(image)
    print_info(image_f)
    print_info(image_t)