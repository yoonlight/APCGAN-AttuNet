from torch.utils.data import DataLoader

from .dataset_BCL import BCLDataset
from .dataset_volker import VolkerDataset
from .dataset_deepcrack import DeepCrackDataset
from .dataset_CFD import CFDDataset


def load_data(dataset_name, data_path, val_data_path, batch_size):
    if dataset_name == 'CFD':
        train_ds = CFDDataset(data_path)
        val_ds = CFDDataset(val_data_path)
    elif dataset_name == 'DeepCrack':
        train_ds = DeepCrackDataset(data_path)
        val_ds = DeepCrackDataset(val_data_path)
    elif dataset_name == 'BCL':
        train_ds = BCLDataset(data_path)
        val_ds = BCLDataset(val_data_path)
    elif dataset_name == 'Volker':
        train_ds = VolkerDataset(data_path)
        val_ds = VolkerDataset(val_data_path)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    train_loader = DataLoader(
        dataset=train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_ds)
    return train_loader, val_loader, len(train_ds)
