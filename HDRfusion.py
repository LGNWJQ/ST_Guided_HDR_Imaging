from models.HDR import HDR_Lightning_System
from dataset.dataset_SIG17 import SIG17_Dataset, SIG17_DataModule
from config_file import set_config
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from config_file import set_config

if __name__ == "__main__":
    config = set_config()
    model = HDR_Lightning_System(config)
    data_module = SIG17_DataModule(
        train_dataset_path=config.training_dataset_path,
        val_dataset_path=config.val_dataset_path,
        patch_size=config.patch_size,
        val_patch_size=config.val_patch_size,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        edge_type=config.edge_type
    )

    trainer = pl.Trainer()
    trainer.predict(
        model=model,
        datamodule=data_module,
        ckpt_path=r'D:\IEEE_SPL\pl_code\SPL_HDR\ckpt\with_ST\epoch=222-val_psnr=35.34.ckpt',
    )