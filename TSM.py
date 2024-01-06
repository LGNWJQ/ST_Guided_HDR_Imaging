import torch
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
import os
import imageio
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from dataset import utils

imageio.plugins.freeimage.download()
imageio.core.util.appdata_dir("imageio")
def print_info(x):
    print('=-' * 20)
    print('shape: ', x.shape)
    print(f'range: [{x.min()}, {x.max()}]')
    print('dtype:', x.dtype)
    print('=-' * 20)

if __name__ == "__main__":
    utils.Mu_Law()