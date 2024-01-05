import torch
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS

"""
DFVDFV
"""

if __name__ == "__main__":
    a = torch.randint(5, (10,))

    print(a)