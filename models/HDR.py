import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
import cv2
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as LPIPS
from torchvision.utils import make_grid
from origin_model import HDR_Network
'''
Feature extraction network and HDR reconstruction network
'''


def Mu_Law(image, mu=5000):
    mu = torch.tensor([mu]).to(image.device)
    return torch.log(1 + image * mu) / torch.log(1 + mu)


class HDR_Lightning_System(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        in_channels = 6 if config.edge_type is None else 7
        self.model = HDR_Network(
            in_channels=in_channels,
            basic_channels=config.basic_channels,
            depths=config.depths,
        )
        self.lr = config.lr
        self.num_epochs = config.num_epochs
        self.example_input_array = (torch.randn(1, in_channels, 256, 256), torch.randn(1, in_channels, 256, 256))

        self.optimizer = Adam(self.parameters(), lr=self.lr)

        self.psnr_metric = PSNR(data_range=1.0, reduction='elementwise_mean')

        self.save_hyperparameters(config)

        self.val_step_hdr = []
        self.val_step_r_hdr = []

    def forward(self, X1, X2):
        return self.model(X1, X2)

    def training_step(self, batch, batch_idx):
        X1 = batch['input1']
        X2 = batch['input2']
        HDR = batch['label']
        r_HDR = self.model(X1, X2)

        HDR_mu = Mu_Law(HDR)
        r_HDR_mu = Mu_Law(r_HDR)
        loss = F.l1_loss(HDR_mu, r_HDR_mu)

        self.log("loss", loss, prog_bar=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return loss

    def predict_step(self, batch, batch_idx):
        X1 = batch['input1']
        X2 = batch['input2']
        r_HDR = self.model(X1, X2)

        HDR = r_HDR.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        cv2.imwrite(f'./lightning_logs/result_st/{batch_idx+1}.hdr', HDR)

    def validation_step(self, batch, batch_idx):
        X1 = batch['input1']
        X2 = batch['input2']
        HDR = batch['label']
        r_HDR = self.model(X1, X2)

        HDR_mu = Mu_Law(HDR)
        r_HDR_mu = Mu_Law(r_HDR)
        val_loss = F.l1_loss(HDR_mu, r_HDR_mu)

        val_psnr = self.psnr_metric(HDR, r_HDR)
        val_psnr_mu = self.psnr_metric(HDR_mu, r_HDR_mu)
        values = {"val_loss": val_loss, "val_psnr": val_psnr, "val_psnr_mu": val_psnr_mu}
        self.log_dict(values, prog_bar=True)

        if len(self.val_step_r_hdr) <= 2:
            self.val_step_r_hdr.append(r_HDR_mu)
        if len(self.val_step_hdr) <= 2:
            self.val_step_hdr.append(HDR_mu)

        return {'r_HDR': r_HDR, 'HDR': HDR}

    def on_validation_epoch_end(self):
        tensorboard = self.logger.experiment
        hdr_list = torch.cat(self.val_step_hdr, dim=0)
        r_hdr_list = torch.cat(self.val_step_r_hdr, dim=0)

        hdr_to_show = make_grid(hdr_list[::2], nrow=4)
        r_hdr_to_show = make_grid(r_hdr_list[::2], nrow=4)

        tensorboard.add_image('hdr_mu', hdr_to_show, global_step=0)
        tensorboard.add_image('r_hdr_mu', r_hdr_to_show, global_step=self.global_step)
        self.val_step_hdr.clear()
        self.val_step_r_hdr.clear()

    def configure_optimizers(self):
        scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.lr / 1e2
        )

        return [self.optimizer], [scheduler]





from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # writer = SummaryWriter('./logs/HDR_Network')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HDR_Network().to(device)
    print("Dual_HDRNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.eval()
    size = [1000, 1500]

    x1 = torch.randn(1, 7, size[0], size[1]).to(device)
    x2 = torch.randn(1, 7, size[0], size[1]).to(device)
    for i in range(20):
        with torch.no_grad():
            out = model(x1, x2)

    # print(out.shape)
    # print(out.dtype)
    # print(out.min(), out.max())
    #
    # writer.add_graph(model, [x1, x2])
    # writer.close()
