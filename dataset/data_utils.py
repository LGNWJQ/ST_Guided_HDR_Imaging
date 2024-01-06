import torch
from torch import nn
import torch.nn.functional as F

def Normalize_neg_one_to_one(img):
    return img * 2 - 1

def Unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

def Gamma_Correction_HtoL(image, gamma=2.2):
    return torch.pow(image, 1 / gamma)


def Gamma_Correction_LtoH(image, gamma=2.2):
    return torch.pow(image, gamma)


def Mu_Law(image, mu=5000):
    return torch.log(1 + image * mu) / torch.log(1 + mu)


class Structure_Tensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_X = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            padding_mode='reflect'
        )
        self.X_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 1, 3)
        self.gradient_X.weight.data = self.X_kernel

        self.gradient_Y = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            padding_mode='reflect'
        )
        self.Y_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 3, 1)
        self.gradient_Y.weight.data = self.Y_kernel

    def forward(self, x):
        # 计算灰度图
        r, g, b = x.unbind(dim=-3)
        gray = (0.2989 * r + 0.587 * g + 0.114 * b)
        gray = gray.unsqueeze(dim=-3) * 255.0

        # 计算梯度
        Ix = self.gradient_X(gray)
        Iy = self.gradient_Y(gray)
        # 二阶导数
        Ix2 = torch.pow(Ix, 2)
        Iy2 = torch.pow(Iy, 2)
        Ixy = Ix * Iy

        # 计算结构张量的行列式K和迹H
        #  Ix2, Ixy
        #  Ixy, Iy2
        H = Ix2 + Iy2
        K = Ix2 * Iy2 - Ixy * Ixy

        # Flat平坦区域：H = 0;
        # Edge边缘区域：H > 0 & & K = 0;
        # Corner角点区域：H > 0 & & K > 0;
        # 浮点数无法精确判断等，

        H_threshold = 100

        Flat = torch.zeros_like(H)
        Flat[H < H_threshold] = 1.0

        Edge = torch.zeros_like(H)
        Edge[(H >= H_threshold) * (K.abs() <= 1e-6)] = 1.0

        Corner = torch.zeros_like(H)
        Corner[(H >= H_threshold) * (K.abs() > 1e-6)] = 1.0

        Gradient_Matrix = torch.cat([Ix2, Ixy, Iy2], dim=1)

        return 1.0 - Flat, F.tanh(Gradient_Matrix)


class Image_Gradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            padding_mode='reflect'
        )
        self.gradient.weight.data = torch.tensor([[-1, -1, -1],
                                                  [-1, 8, -1],
                                                  [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, image):
        r, g, b = image.unbind(dim=-3)
        gray = (0.2989 * r + 0.587 * g + 0.114 * b)
        gray = gray.unsqueeze(dim=-3)
        grad = self.gradient(gray)

        return grad
