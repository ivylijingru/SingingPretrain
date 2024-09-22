import torch
import torch.nn as nn
import torch.nn.functional as F


class CombineLoss(nn.Module):
    def __init__(
        self,
        onset_pos_weight: float = 15.0,
    ):
        super().__init__()
        self.onset_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([onset_pos_weight,]), reduction="none")
        self.silence_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.pitch_class_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.octave_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, output_arr, target_arr):
        loss_dict = dict()

        onset_pred = output_arr[:, :, 0].flatten()
        onset_gt = target_arr[:, :, 0].flatten()
        onset_loss = self.onset_loss_fn(onset_pred, onset_gt)
        loss_dict["loss/onset"] = onset_loss.mean()
        
        silence_pred = output_arr[:, :, 1].flatten()
        silence_gt = target_arr[:, :, 1].flatten()
        silence_loss = self.silence_loss_fn(silence_pred, silence_gt)
        loss_dict["loss/silence"] = silence_loss.mean()

        octave_pred = output_arr[:, :, 2:7].flatten(end_dim=-2)
        octave_gt = target_arr[:, :, 2].flatten().long()
        octave_loss = self.octave_loss_fn(octave_pred, octave_gt)
        loss_dict["loss/octave"] = octave_loss.mean()

        pitch_class_pred = output_arr[:, :, 7:].flatten(end_dim=-2)
        pitch_class_gt = target_arr[:, :, 3].flatten().long()
        pitch_class_loss = self.pitch_class_loss_fn(pitch_class_pred, pitch_class_gt)
        loss_dict["loss/pitch"] = pitch_class_loss.mean()

        return loss_dict
