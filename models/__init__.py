from typing import Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel

from .basemodels import get_base_model
from .loss_fn import get_loss_fn


class SVTDownstreamModel(pl.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.optim_cfg = configs["optim"]
        self.model = get_base_model(configs["mlp"])
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
        self.freeze_epoch = configs["finetune"]["freeze_epoch"]
        self.loss_fn = get_loss_fn(configs["loss"])

    def on_train_epoch_start(self):
        # 冻结 mert_model 参数直到 freeze_epoch 结束
        self.mert_model.config.mask_time_prob = 0.0
        if self.current_epoch < self.freeze_epoch:
            for param in self.mert_model.parameters():
                param.requires_grad = False
        else:
            for param in self.mert_model.parameters():
                param.requires_grad = False
            for param in self.mert_model.encoder.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx, optimizer_idx) -> Any:
        loss_dict, _ = self.common_step(batch)

        self.log("lr_mert", self.optimizers()[0].param_groups[0]["lr"])
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
        # mert = batch["mert"]
        inputs = batch["inputs"]
        y = batch["y"]
        
        loss_dict = dict()
        # must set eval here; if in init will automatically set to training mode
        outputs = self.mert_model(**inputs, output_hidden_states=True)
        # outputs: [bs, time, feature_shape]
        # we want hidden states to be: [bs, n_channels, time, feature_shape]
        all_layer_hidden_states = torch.stack(outputs.hidden_states, dim=1)
        mert = all_layer_hidden_states

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

    def inference_step(self, batch):
        # mert = batch["mert"]
        inputs = batch["inputs"]
        
        # must set eval here; if in init will automatically set to training mode
        outputs = self.mert_model(**inputs, output_hidden_states=True)
        # outputs: [bs, time, feature_shape]
        # we want hidden states to be: [bs, n_channels, time, feature_shape]
        all_layer_hidden_states = torch.stack(outputs.hidden_states, dim=1)
        mert = all_layer_hidden_states

        model_output = self.model(mert)

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
        optimizer_cfg_mert = self.optim_cfg["optimizer_mert"]
        optimizer_cfg_mlp = self.optim_cfg["optimizer_mlp"]
        scheduler_cfg = self.optim_cfg["scheduler"]

        optimizer_mert = torch.optim.__dict__.get(optimizer_cfg_mert["name"])(self.mert_model.encoder.parameters(), **optimizer_cfg_mert["args"])
        optimizer_mlp = torch.optim.__dict__.get(optimizer_cfg_mlp["name"])(self.model.parameters(), **optimizer_cfg_mlp["args"])
        scheduler_mert = torch.optim.lr_scheduler.__dict__.get(scheduler_cfg["name"])(optimizer_mert, **scheduler_cfg["args"])
        scheduler_mlp = torch.optim.lr_scheduler.__dict__.get(scheduler_cfg["name"])(optimizer_mlp, **scheduler_cfg["args"])

        return (
            dict(
                optimizer=optimizer_mert,
                lr_scheduler=dict(
                    scheduler=scheduler_mert,
                    monitor=scheduler_cfg["monitor"],
            )),
            dict(
                optimizer=optimizer_mlp,
                lr_scheduler=dict(
                    scheduler=scheduler_mlp,
                    monitor=scheduler_cfg["monitor"],
            )),
        )