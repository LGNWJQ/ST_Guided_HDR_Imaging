import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import dataset.data_utils as utils

import os
import imageio
import numpy as np

imageio.plugins.freeimage.download()
imageio.core.util.appdata_dir("imageio")


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

    def __init__(self, dataset_path, patch_size, isTraining=True, edge_type=None):
        super().__init__()
        self.dataset_path = dataset_path
        data_folder_list = os.listdir(self.dataset_path)
        self.data_folder_list = [folder for folder in data_folder_list
                                 if os.path.exists(os.path.join(self.dataset_path, folder, 'HDRImg.hdr'))]
        # 数据增强
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
                "image_tmap": "image",
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
                "image_tmap": "image",
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
        data_folder = os.path.join(self.dataset_path, self.data_folder_list[index])
        file_list = sorted(os.listdir(data_folder))

        # 获取各文件路径
        ldr1_path = os.path.join(data_folder, file_list[0])
        ldr2_path = os.path.join(data_folder, file_list[1])
        hdr_path = os.path.join(data_folder, file_list[2])
        hdr_tmap_path = os.path.join(data_folder, file_list[3])
        ev_file_path = os.path.join(data_folder, file_list[4])

        # 读取为numpy, 统一数据类型为0-1的float32
        ldr_image1 = imageio.v2.imread(ldr1_path).astype("float32") / 65535.0
        ldr_image2 = imageio.v2.imread(ldr2_path).astype("float32") / 65535.0
        hdr_image = imageio.v2.imread(hdr_path, format='HDR-FI')
        hdr_tmap_image = imageio.v2.imread(hdr_tmap_path).astype("float32") / 65535.0
        # expo_value = np.loadtxt(ev_file_path, dtype=np.float32)
        # expo_value = torch.from_numpy(expo_value)

        # 数据处理：统一数据类型为0-1的float32，数据增强，转化为tensor
        augmentations = self.transform(
            image=hdr_image,
            image1=ldr_image1,
            image2=ldr_image2,
            image_tmap=hdr_tmap_image,
        )
        ldr_image1 = augmentations['image1']
        ldr_image2 = augmentations['image2']
        hdr_image = augmentations['image']
        hdr_tmap_image = augmentations['image_tmap']

        # 对输入数据进行处理：将ldr伽马映射为hdr，计算边缘检测图
        hdr_from_ldr1 = utils.Gamma_Correction_LtoH(ldr_image1)
        hdr_from_ldr2 = utils.Gamma_Correction_LtoH(ldr_image2)
        X1 = torch.cat([ldr_image1, hdr_from_ldr1], dim=-3)
        X2 = torch.cat([ldr_image2, hdr_from_ldr2], dim=-3)

        # 计算第七个边缘通道
        if self.edge_type is not None:
            if self.edge_type == "ST":
                st_edge1, _ = self.ST(ldr_image1)
                st_edge2, _ = self.ST(ldr_image2)
                X1 = torch.cat([X1, st_edge1], dim=-3)
                X2 = torch.cat([X2, st_edge2], dim=-3)

            if self.edge_type == "IG":
                ig_edge1 = self.IG(ldr_image1).detach()
                ig_edge2 = self.IG(ldr_image2).detach()
                X1 = torch.cat([X1, ig_edge1], dim=-3)
                X2 = torch.cat([X2, ig_edge2], dim=-3)

        sample = {
            'ldr1': X1,
            'ldr2': X2,
            'hdr': hdr_image,
            'hdr_tmap': hdr_tmap_image
        }

        return sample


from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.nn import functional as F


def print_info(x):
    print('=-' * 20)
    print('shape: ', x.shape)
    print(f'range: [{x.min()}, {x.max()}]')
    print('dtype:', x.dtype)
    print('=-' * 20)


if __name__ == "__main__":
    dataset_path = "D:/BASIC_FILE/DT/My_Dataset/TEST/"
    patch_size = 512
    isTraining = False
    edge_type = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ST = utils.Structure_Tensor().to(device)
    # IG = utils.Image_Gradient().to(device)

    dataset = SIG17_Dataset(dataset_path, patch_size=patch_size, isTraining=isTraining, edge_type=edge_type)
    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )
    for batch in loader:
        ldr1, ldr2 = batch['ldr1'].to(device), batch['ldr2'].to(device)
        hdr, hdr_tmap = batch['hdr'].to(device), batch['hdr_tmap'].to(device)

        print_info(ldr1)
        print_info(ldr2)
        print_info(hdr)
        print_info(hdr_tmap)

        save_image(ldr1[:, :3], 'batch/ldr1.tif')
        save_image(ldr2[:, :3], 'batch/ldr2.tif')
        save_image(hdr, 'batch/hdr.tif')
        save_image(torch.pow(hdr, 1/2.2), 'batch/hdr_mu.tif')
        save_image(hdr_tmap, 'batch/hdr_tmap.tif')

        break
