import torch.nn as nn
import torch.nn.functional as F


class CombineLoss(nn.Module):
    def __init__(
        self,
        onset_weight: float = 15.0,
        silence_weight: float = 1.0,
    ):
        super().__init__()
        self.onset_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.silence_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.pitch_class_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.octave_loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.onset_weight = onset_weight
        self.silence_weight = silence_weight

    def forward(self, output_arr, target_arr):
        onset_pred = output_arr[:, :, 0].flatten()
        onset_gt = target_arr[:, :, 0].flatten()
        onset_loss = self.onset_loss_fn(onset_pred, onset_gt)
        
        silence_pred = output_arr[:, :, 1].flatten()
        silence_gt = target_arr[:, :, 1].flatten()
        silence_loss = self.silence_loss_fn(silence_pred, silence_gt)

        octave_pred = output_arr[:, :, 2:7].flatten(end_dim=-2)
        octave_gt = target_arr[:, :, 2].flatten().long()
        octave_loss = self.octave_loss_fn(octave_pred, octave_gt)

        pitch_class_pred = output_arr[:, :, 7:].flatten(end_dim=-2)
        pitch_class_gt = target_arr[:, :, 3].flatten().long()
        pitch_class_loss = self.pitch_class_loss_fn(pitch_class_pred, pitch_class_gt)

        return self.onset_weight * onset_loss + self.silence_weight * silence_loss + octave_loss + pitch_class_loss
