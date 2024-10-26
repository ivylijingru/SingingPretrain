HOME_PATH = "/home/jli3268" # path where you cloned musicfm

import os
import sys
import torch

sys.path.append(HOME_PATH)
from musicfm.model.musicfm_25hz import MusicFM25Hz

# dummy audio (30 seconds, 24kHz)
wav = (torch.rand(4, 24000 * 30) - 0.5) * 2

# load MusicFM
musicfm = MusicFM25Hz(
    is_flash=False,
    stat_path=os.path.join(HOME_PATH, "musicfm", "data", "msd_stats.json"),
    model_path=os.path.join(HOME_PATH, "musicfm", "data", "pretrained_msd.pt"),
)

# to GPUs
wav = wav.cuda()
musicfm = musicfm.cuda()

# get embeddings
musicfm.eval()
_, hidden_states = musicfm.get_predictions(wav)
all_hidden_states = torch.stack(hidden_states, dim=1)
