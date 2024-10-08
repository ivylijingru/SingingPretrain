from typing import Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .basemodels import get_base_model
from .loss_fn import get_loss_fn


class SVTDownstreamModel(pl.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.optim_cfg = configs["optim"]
        self.model = get_base_model(configs["mlp"])
        self.loss_fn = get_loss_fn(configs["loss"])

    def training_step(self, batch, batch_idx) -> Any:
        loss_dict, _ = self.common_step(batch)

        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"])
        self.log_dict_prefix(loss_dict, "train")

        # self.train_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

        return loss_dict["loss/total"]

    def validation_step(self, batch, batch_idx) -> Any:
        loss_dict, _ = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "val")
        
        # self.val_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

        return loss_dict["loss/total"]

    def test_step(self, batch, batch_idx):
        loss_dict, _ = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "test")

        # self.test_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

    def common_step(self, batch):
        mert = batch["mert"]
        y = batch["y"]
        
        loss_dict = dict()

        model_output = self.model(mert)
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

    def log_dict_prefix(self, d, prefix):
        for k, v in d.items():
            self.log("{}/{}".format(prefix, k), v)

    def configure_optimizers(self) -> Any:
        optimizer_cfg = self.optim_cfg["optimizer"]
        scheduler_cfg = self.optim_cfg["scheduler"]

        optimizer = torch.optim.__dict__.get(optimizer_cfg["name"])(self.parameters(), **optimizer_cfg["args"])
        scheduler = torch.optim.lr_scheduler.__dict__.get(scheduler_cfg["name"])(optimizer, **scheduler_cfg["args"])
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor=scheduler_cfg["monitor"],
            ))
