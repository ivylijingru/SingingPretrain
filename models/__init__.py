from typing import Any
import sys
import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoModel

from .basemodels import get_base_model
from .loss_fn import get_loss_fn

HOME_PATH = "/home/jli3268" # path where you cloned musicfm
sys.path.append(HOME_PATH)
from musicfm.model.musicfm_25hz import MusicFM25Hz

class SVTDownstreamModel(pl.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.optim_cfg = configs["optim"]
        self.model = get_base_model(configs["mlp"])
        self.musicfm_model = MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(HOME_PATH, "musicfm", "data", "msd_stats.json"),
            model_path=os.path.join(HOME_PATH, "musicfm", "data", "pretrained_msd.pt"),
        )
        self.freeze_epoch = configs["finetune"]["freeze_epoch"]
        self.loss_fn = get_loss_fn(configs["loss"])

    def on_train_epoch_start(self):
        # 冻结 musicfm_model 参数直到 freeze_epoch 结束
        # if self.current_epoch < self.freeze_epoch:
        #     for param in self.musicfm_model.parameters():
        #         param.requires_grad = False
        # else:
        for param in self.musicfm_model.parameters():
            param.requires_grad = False
        for param in self.musicfm_model.conformer.parameters():
            param.requires_grad = True
        if self.current_epoch < self.freeze_epoch:
            self.musicfm_model.conformer.eval()
        else:
            self.musicfm_model.conformer.train()

    def training_step(self, batch, batch_idx, optimizer_idx) -> Any:
        loss_dict, _ = self.common_step(batch)

        self.log("lr_musicfm", self.optimizers()[0].param_groups[0]["lr"])
        self.log("lr_mlp", self.optimizers()[1].param_groups[0]["lr"])
        self.log_dict_prefix(loss_dict, "train")

        return loss_dict["loss/total"]

    def validation_step(self, batch, batch_idx) -> Any:
        loss_dict, _ = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "val")

        return loss_dict["loss/total"]

    def test_step(self, batch, batch_idx):
        loss_dict, _ = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "test")

    def common_step(self, batch):
        inputs = batch["inputs"]
        y = batch["y"]
        
        loss_dict = dict()
        # must set eval here; if in init will automatically set to training mode
        _, hidden_states = self.musicfm_model.get_predictions(inputs)
        musicfm_emb = torch.stack(hidden_states, dim=1)

        model_output = self.model(musicfm_emb)
        loss_dict = self.loss_fn(model_output, y)

        total_loss = 0
        for loss_key in loss_dict.keys():
            total_loss += loss_dict[loss_key]
        loss_dict["loss/total"] = total_loss

        logic_dict = dict()
        logic_dict["onset"] = model_output[:, :, 0]
        logic_dict["silence"] = model_output[:, :, 1]
        logic_dict["octave"] = model_output[:, :, 2:7]
        logic_dict["pitch"] = model_output[:, :, 7:20]

        return loss_dict, logic_dict

    def inference_step(self, batch):
        inputs = batch["inputs"]
        
        # must set eval here; if in init will automatically set to training mode
        _, hidden_states = self.musicfm_model.get_predictions(inputs)
        musicfm_emb = torch.stack(hidden_states, dim=1)

        model_output = self.model(musicfm_emb)

        logic_dict = dict()
        logic_dict["onset"] = model_output[:, :, 0]
        logic_dict["silence"] = model_output[:, :, 1]
        logic_dict["octave"] = model_output[:, :, 2:7]
        logic_dict["pitch"] = model_output[:, :, 7:20]

        return logic_dict

    def log_dict_prefix(self, d, prefix):
        for k, v in d.items():
            self.log("{}/{}".format(prefix, k), v)

    def configure_optimizers(self) -> Any:
        # here we define two optimizers with different learning rate
        optimizer_cfg_musicfm = self.optim_cfg["optimizer_musicfm"]
        optimizer_cfg_mlp = self.optim_cfg["optimizer_mlp"]
        scheduler_cfg = self.optim_cfg["scheduler"]

        optimizer_musicfm = torch.optim.__dict__.get(optimizer_cfg_musicfm["name"])(self.musicfm_model.conformer.parameters(), **optimizer_cfg_musicfm["args"])
        optimizer_mlp = torch.optim.__dict__.get(optimizer_cfg_mlp["name"])(self.model.parameters(), **optimizer_cfg_mlp["args"])
        scheduler_musicfm = torch.optim.lr_scheduler.__dict__.get(scheduler_cfg["name"])(optimizer_musicfm, **scheduler_cfg["args"])
        scheduler_mlp = torch.optim.lr_scheduler.__dict__.get(scheduler_cfg["name"])(optimizer_mlp, **scheduler_cfg["args"])

        # # Define a custom lambda function to control learning rate based on epoch number
        # def lr_lambda(epoch):
        #     if epoch < self.freeze_epoch:
        #         return 1.0  # Keep initial lr (3e-5) for the first 90 epochs
        #     else:
        #         return 5e-5 / 3e-3  # Change to 5e-5 after 90 epochs
        
        # scheduler_mlp = LambdaLR(optimizer_mlp, lr_lambda)
        
        # Define a custom lambda function to control learning rate based on epoch number
        def lr_lambda(epoch):
            if epoch < self.freeze_epoch:
                return 0.0  # Keep initial lr (3e-5) for the first 90 epochs
            else:
                return 1.0  # Change to 5e-5 after 90 epochs
        
        scheduler_musicfm = LambdaLR(optimizer_musicfm, lr_lambda)

        return (
            dict(
                optimizer=optimizer_musicfm,
                lr_scheduler=dict(
                    scheduler=scheduler_musicfm,
                    monitor=scheduler_cfg["monitor"],
            )),
            dict(
                optimizer=optimizer_mlp,
                lr_scheduler=dict(
                    scheduler=scheduler_mlp,
                    monitor=scheduler_cfg["monitor"],
            )),
        )