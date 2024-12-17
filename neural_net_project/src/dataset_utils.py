import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from PIL import Image
import configs.config

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None) -> None:
        self.labels_df = pd.read_csv(labels_csv)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # Ensure image is RGB
        label = self.labels_df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def CustomDataLoader(dataset, training_portion, batch_size=32, shuffle=True):
    size = len(dataset)
    train_size = int(training_portion * size)
    val_size = size - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size, shuffle)
    val_loader = DataLoader(val_set, batch_size, shuffle)

    return train_loader, val_loader
