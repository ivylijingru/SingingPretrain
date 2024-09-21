import torch.utils.data as Data
import pytorch_lightning as pl

from .dataset import TranscriptionDataset


class TranscriptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_manifest_path,
        test_manifest_path,
        slice_sec,
        token_rate,
        batch_size,
        train_shuffle,
        num_workers,
    ) -> None:
        super().__init__()

        self.train_dataset = TranscriptionDataset(train_manifest_path, slice_sec, token_rate)
        self.test_dataset = TranscriptionDataset(test_manifest_path, slice_sec, token_rate)

        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        return Data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return Data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return Data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


