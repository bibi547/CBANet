import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn as nn

# from models.teethgroup import TeethGroup
from models.cbanet import CBANet
from data.st_data import Teeth3DS
from utils.loss import CPLoss, match_loss


def find_peak(heatmap, xs):
    return torch.stack([xs[idx][:, max_idx].T for idx, max_idx in enumerate(torch.argmax(heatmap, axis=2))])


def criterion(a, b):
    return torch.norm(a - b, dim=-1).mean()


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = CBANet(args)
        self.bmap_loss = nn.CrossEntropyLoss()
        self.dmap_loss = nn.MSELoss()
        self.mask_loss = CPLoss(args)

    def forward(self, x):
        return self.net(x, self.current_epoch)

    def infer(self, x):
        out_bmap, out_dmap, sp_masks, sp_probs, score = self.net.inference(x)
        return out_bmap, out_dmap, sp_masks, sp_probs, score

    def training_step(self, batch, _):
        x, y, bmap, dmap, ins_masks, ins_labels = batch
        out_bmap, out_dmap, sp_masks, sp_probs, score, all_idx = self(x)

        bl = self.bmap_loss(out_bmap, bmap)
        dl = self.dmap_loss(out_dmap, dmap) * 5
        gl = 0.0
        if self.current_epoch <= 19:
            loss = bl + dl
        else:
            gl, lc, lm, ls = self.mask_loss(ins_masks, ins_labels, sp_masks, sp_probs, score, all_idx)
            loss = bl + dl + gl
        self.log('tl/bl', bl, True)
        self.log('tl/dl', dl, True)
        self.log('tl/gl', gl, True)
        self.log('loss', loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, _):
        x, y, bmap, dmap, ins_masks, ins_labels = batch
        out_bmap, out_dmap, sp_masks, sp_probs, score, all_idx = self(x)

        bl = self.bmap_loss(out_bmap, bmap)
        dl = self.dmap_loss(out_dmap, dmap) * 5
        gl = 0.0
        if self.current_epoch <= 19:
            loss = bl + dl
        else:
            gl, lc, lm, ls = self.mask_loss(ins_masks, ins_labels, sp_masks, sp_probs, score, all_idx)
            loss = bl + dl + gl
        self.log('vl/bl', bl, True)
        self.log('vl/dl', dl, True)
        self.log('vl/gl', gl, True)
        self.log('val_loss', loss, True)

    def test_step(self, batch, _):
        x, y, bmap, dmap, ins_masks, ins_labels = batch
        out_bmap, out_dmap, sp_masks, sp_probs, score, all_idx = self(x)

        bl = self.bmap_loss(out_bmap, bmap)
        dl = self.dmap_loss(out_dmap, dmap) * 5
        gl, lc, lm, ls = self.mask_loss(ins_masks, ins_labels, sp_masks, sp_probs, score, all_idx)
        loss = bl + dl + gl
        self.log('test_loss/ce', bl, True)
        self.log('test_loss/l2', dl, True)
        self.log('test_loss/gl', gl, True)
        self.log('test_loss', loss, True)

    def configure_optimizers(self):
        args = self.hparams.args
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.train_file, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.val_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(Teeth3DS(args, args.test_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)
