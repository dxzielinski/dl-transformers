import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

class SpeechCommandsMelDataset(Dataset):
    def __init__(self, root_dir, label_map=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = label_map or self._create_label_map()

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for file in os.listdir(label_path):
                if file.endswith(".pt"):
                    self.samples.append((os.path.join(label_path, file), self.label_map[label]))

    def _create_label_map(self):
        labels = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel_tensor = torch.load(path)  # [1, 80, T] expected
        if self.transform:
            mel_tensor = self.transform(mel_tensor)
        return mel_tensor, label


def get_dataloaders(base_dir="data-mel-spectrograms", batch_size=32, num_workers=2, label_map=None):
    train_dataset = SpeechCommandsMelDataset(os.path.join(base_dir, "train"), label_map)
    val_dataset = SpeechCommandsMelDataset(os.path.join(base_dir, "val"), label_map)
    test_dataset = SpeechCommandsMelDataset(os.path.join(base_dir, "test"), label_map)

    label_map = train_dataset.label_map  # sync label map if not passed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, label_map
