# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn

import os
from skimage import io, transform
import tqdm
from tqdm.auto import tqdm
import zipfile

from cdataset import CustomDataset


if __name__ == '__main__':

    # (HYPER)PARAMETERS
    BATCH_SIZE = 8
    NUM_WORKERS = os.cpu_count()
    RESIZE = 64

    if not os.path.exists('./../DataProcessed/data_256'):
        # Step 1: Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile('./../data_256.zip', 'r') as zip_ref:
            zip_ref.extractall('./../DataProcessed')

    # DATASET DF SETUP
    dataset = pd.read_csv("./../DataMeta/MAMe_dataset.csv")
    labels = pd.read_csv("./../DataMeta/MAMe_labels.csv", header=None)
    toy_data = pd.read_csv("./../DataMeta/MAMe_toy_dataset.csv")

    important = dataset[["Image file", "Subset", "Medium"]]
    important = important.rename(columns={"Medium": "label"})
    important = important.rename(columns={"Image file": "file_path"})
    important["file_path"] = important["file_path"].apply(lambda x: "./../DataProcessed/data_256/" + str(x))

    print("Mapping labels...")
    label_mapper = labels.to_dict()[1]
    label_mapper = {v: k for k, v in label_mapper.items()}
    important["label"] = important["label"].map(label_mapper)
    important = important.dropna()
    important["label"].astype(int)
    #number_classes = len(important["label"].drop_duplicates())

    #important = pd.get_dummies(important, columns=["label"], prefix="label_")
    #labels = [x for x in important.columns if "label" in x]

    print("Creating train, val, test dfs...")

    train_df = important[important['Subset'] == 'train']
    val_df = important[important['Subset'] == 'val']
    test_df = important[important['Subset'] == 'test']

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train df of length {len(train_df)}, validation df of length {len(val_df)} and test df of length {len(test_df)} is created.")
    print(f"Creating dataset and dataLoader with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

    #todo add more transformations for train and add normalization to both

    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(10),  # Rotate the image by up to 10 degrees
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Random crop with resizing to 64x64
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Random color jitter
    ]

    train_transform = transforms.Compose(augmentations + [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.3460, 0.3094, 0.2435], std=[0.2309, 0.2221, 0.2056])
        ])
    train_dataset = CustomDataset(train_df, train_transform)
    train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)


    val_transform = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()])
    val_dataset = CustomDataset(val_df, val_transform)
    val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device is {device}.')



    batch1, label1 = next(iter(train_loader))

    import matplotlib.pyplot as plt

    for pic in batch1:
        data = pic.squeeze().permute(1, 2, 0)
        plt.imshow(data, cmap="gray", interpolation="none")
        plt.show()








