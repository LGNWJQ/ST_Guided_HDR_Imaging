from models.HDR import HDR_Lightning_System
from dataset.dataset_SIG17 import SIG17_DataModule, SIG17_Dataset
from config_file import set_config

import torch
torch.set_float32_matmul_precision('high')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelSummary, DeviceStatsMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(seed=3407)
if __name__ == "__main__":
    config = set_config()
    logger = TensorBoardLogger(
        save_dir="./lightning_logs", name=config.exp_name,
        log_graph=True,
    )

    ES_callback = EarlyStopping(monitor="val_psnr", min_delta=0.00, patience=50, verbose=False, mode="max")
    ckpt_callback = ModelCheckpoint(
        dirpath=f'./ckpt/{config.exp_name}',
        filename='{epoch}-{val_psnr:.2f}',
        monitor='val_psnr',
        mode='max',
        save_top_k=5,
        save_last=False
    )
    tqdm_callback = TQDMProgressBar(refresh_rate=2, process_position=0)
    MS_callback = ModelSummary(max_depth=3)
    DS_callback = DeviceStatsMonitor()
    callback_list = [ES_callback, MS_callback, DS_callback, ckpt_callback, tqdm_callback]

    # profiler = AdvancedProfiler(dirpath="logs", filename="perf_logs")

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

    # train model
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        min_epochs=config.min_epochs,
        log_every_n_steps=1,
        callbacks=callback_list,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        logger=logger,
        # profiler=profiler,
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=config.ckpt_path,
    )
