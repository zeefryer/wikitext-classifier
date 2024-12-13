"""Custom dataset classes."""

from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, text, labels, transform=None):
        self.text = text
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.text[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
