import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import dataset.data_utils as utils

import os
import numpy as np
import imageio

imageio.plugins.freeimage.download()
imageio.core.util.appdata_dir("imageio")  # 下载失败时前往官网下载文件放置到该文件夹下的freeimage文件夹内


class SIG17_DataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset_path,
            val_dataset_path,
            patch_size,
            val_patch_size,
            batch_size,
            val_batch_size,
            num_workers,
            edge_type=None
    ):
        super().__init__()
        self.predict_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.patch_size = patch_size
        self.val_patch_size = val_patch_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.edge_type = edge_type

        in_channels = 6 if edge_type is None else 7
        self.dims = (in_channels, patch_size, patch_size)

    def setup(self, stage=None):
        self.train_dataset = SIG17_Dataset(
            train_dataset_path=self.train_dataset_path,
            patch_size=self.patch_size,
            isTraining=True,
            edge_type=self.edge_type
        )
        self.val_dataset = SIG17_Dataset(
            train_dataset_path=self.val_dataset_path,
            patch_size=self.val_patch_size,
            isTraining=False,
            edge_type=self.edge_type
        )
        self.predict_dataset = SIG17_Dataset(
            train_dataset_path=self.val_dataset_path,
            patch_size=0,
            isTraining=False,
            edge_type=self.edge_type
        )
        self.predict_dataset.transform = A.Compose(
            [
                ToTensorV2(),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
            },
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )


class SIG17_Dataset(Dataset):
    """
    参数：
    - dataset_path: 训练集的路径
    - patch_size:   训练时图像裁剪大小
    - isTraining:   是否是训练集
    - edge_type:    第七通道类型
    """

    def __init__(self, train_dataset_path, patch_size, isTraining=True, edge_type=None):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        # 过滤无效文件or文件夹
        data_folder_list = os.listdir(self.train_dataset_path)
        self.data_folder_list = [folder for folder in data_folder_list
                                 if os.path.exists(os.path.join(self.train_dataset_path, folder, 'HDRImg.hdr'))]
        # 数据增强
        self.ToFloat32 = A.ToFloat(max_value=65535.0)
        self.train_transform = A.Compose(
            [
                A.RandomCrop(width=patch_size, height=patch_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
            },
        )
        self.test_transform = A.Compose(
            [
                A.CenterCrop(width=patch_size, height=patch_size),
                ToTensorV2(),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
            },
        )
        self.transform = self.train_transform if isTraining else self.test_transform

        self.edge_type = edge_type
        if edge_type is not None:
            if edge_type == "ST":
                self.ST = utils.Structure_Tensor()
            if edge_type == "IG":
                self.IG = utils.Image_Gradient()

    def __len__(self):
        return len(self.data_folder_list)

    def __getitem__(self, index):
        data_folder = os.path.join(self.train_dataset_path, self.data_folder_list[index])
        file_list = sorted(os.listdir(data_folder))
        # 获取各文件路径
        ldr_image1_path = os.path.join(data_folder, file_list[0])
        ldr_image2_path = os.path.join(data_folder, file_list[1])
        hdr_image_path = os.path.join(data_folder, file_list[2])
        ev_file_path = os.path.join(data_folder, file_list[3])
        # 读取为numpy, 统一数据类型为0-1的float32
        ldr_image1 = imageio.v2.imread(ldr_image1_path)
        ldr_image2 = imageio.v2.imread(ldr_image2_path)
        hdr_image = imageio.v2.imread(hdr_image_path, format='HDR-FI')
        # expo_value = np.loadtxt(ev_file_path, dtype=np.float32)
        # expo_value = torch.from_numpy(expo_value)

        # 数据处理：统一数据类型为0-1的float32，数据增强，转化为tensor
        ldr_image1 = self.ToFloat32(image=ldr_image1)['image']
        ldr_image2 = self.ToFloat32(image=ldr_image2)['image']
        augmentations = self.transform(
            image=hdr_image,
            image1=ldr_image1,
            image2=ldr_image2,
        )
        ldr_image1_t = augmentations['image1']
        ldr_image2_t = augmentations['image2']
        hdr_image_t = augmentations['image']

        # 对输入数据进行处理：将ldr伽马映射为hdr，计算边缘检测图
        hdr_from_ldr1 = utils.Gamma_Correction_LtoH(ldr_image1_t)
        hdr_from_ldr2 = utils.Gamma_Correction_LtoH(ldr_image2_t)
        X1 = torch.cat([ldr_image1_t, hdr_from_ldr1], dim=-3)
        X2 = torch.cat([ldr_image2_t, hdr_from_ldr2], dim=-3)

        # 计算第七个边缘通道
        if self.edge_type is not None:
            if self.edge_type == "ST":
                st_edge1, _ = self.ST(ldr_image1_t)
                st_edge2, _ = self.ST(ldr_image2_t)
                X1 = torch.cat([X1, st_edge1], dim=-3)
                X2 = torch.cat([X2, st_edge2], dim=-3)

            if self.edge_type == "IG":
                ig_edge1 = self.IG(ldr_image1_t).detach()
                ig_edge2 = self.IG(ldr_image2_t).detach()
                X1 = torch.cat([X1, ig_edge1], dim=-3)
                X2 = torch.cat([X2, ig_edge2], dim=-3)

        sample = {
            'input1': X1,
            'input2': X2,
            'label': hdr_image_t
        }

        return sample


from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.nn import functional as F

if __name__ == "__main__":
    dataset_path = r"D:\IEEE_SPL\HDR_DATA\TRAIN"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ST = utils.Structure_Tensor().to(device)
    IG = utils.Image_Gradient().to(device)

    dataset = SIG17_Dataset(dataset_path, patch_size=(1000, 1000), isTraining=False, edge_type='IG')
    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=2,
        shuffle=False,
    )
    for batch in loader:
        ldr1, ldr2, hdr = batch['input1'].to(device), batch['input2'].to(device), batch['label'].to(device)
        image = ldr1[:, 6:, :, :]

        print(ldr2.shape)
        save_image(image, 'debug_file/final_batch/ig_st_edge.png')

        # ldr1_st, matrix = ST(image)
        # ldr1_ig = IG(image)

        # matrix = F.tanh(matrix)

        # print('ldr dtype:', ldr1_st.dtype, ldr1_ig.dtype)
        # print('st_range:', ldr1_st.min(), ldr1_st.max())
        # print('ig_range:', ldr1_ig.min(), ldr1_ig.max())
        # print('matrix:', matrix.min(), matrix.max())
        # print('shape:', ldr1_st.shape, ldr1_ig.shape)
        # print('结构张量的梯度函数：', matrix.grad_fn)
        # print('结构张量平面的梯度函数：', ldr1_st.grad_fn)

        # save_image(ldr1_st, 'edge_batch/ldr1_st.png')
        # save_image(ldr1_ig, 'edge_batch/ldr1_ig.png')
        # save_image(matrix, 'edge_batch/ldr1_matrix.png')

        # print('type:', type(ldr1), type(hdr))
        # print('ldr dynamic:', ldr1.min(), 'to', ldr2.max())
        # print('hdr dynamic:', hdr.min(), 'to', hdr.max())
        # print('data type:', ldr1.dtype, hdr.dtype)
        # print('shape:', ldr1.shape, hdr.shape)
        # save_image(ldr1, 'gamma_batch/ldr1.tif')
        # save_image(ldr2, 'gamma_batch/ldr2.tif')
        # save_image(hdr, 'gamma_batch/hdr.tif')
        # save_image(torch.pow(hdr, 1/2.2), 'gamma_batch/hdr_to_ldr.tif')

        break

    # image = dataset.__getitem__(1)

    # ldr = image * 65535
    # # ldr = np.power(image, 1/2.2) * 65535
    # ldr = ldr.astype(np.uint16)
    # imageio.v2.imwrite('./hdr.tif', ldr)

    # ldr = image.astype(np.float32) / 65535
    # print(type(ldr))
    # print(ldr.max(), ldr.min())
    # print(ldr.shape)
    # print(ldr.dtype, image.dtype)
    # ldr = np.power(ldr, 2.2) * 65535
    # ldr = ldr.astype(np.uint16)
    # print(ldr.max(), ldr.min())
    # imageio.v2.imwrite('./ldr_to_hdr.tif', ldr)
