import pickle

import numpy as np
import tensorflow as tf
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cdataset import CustomDataset
device = "cuda"

def full_network_embedding(model, df, batch_size):
    model.to(device)

    val_transform = torchvision.transforms.Compose([
        transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = CustomDataset(df, val_transform)
    val_loader = DataLoader(df, batch_size=batch_size, shuffle=False)

    model.eval()
    # Turn on inference mode
    with torch.no_grad():
        for X, y in val_dataset:
            # Send data to target device
            #TODO: kein plan wie man das macht und kein Bock mehr :D

            X = torch.Tensor(X)
            y = torch.Tensor(y)
            y = y.long()
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred = model(X)  # raw logits
            print(type(val_pred))


import os

from sklearn.metrics import classification_report, confusion_matrix


if __name__ == '__main__':
    # This shows an example of calling the full_network_embedding method using
    # the VGG16 architecture pretrained on ILSVRC2012 (aka ImageNet), as
    # provided by the keras package. Using any other pretrained CNN
    # model is straightforward.

    # Load model
    img_width, img_height = 256, 256
    model = torch.load('model_resnet18')


    import pandas as pd

    dataset = pd.read_csv("./../../../../../DataMeta/MAMe_dataset.csv")
    labels = pd.read_csv("./../../../../../DataMeta/MAMe_labels.csv", header=None)
    toy_data = pd.read_csv("./../../../../../DataMeta/MAMe_toy_dataset.csv")

    important = dataset[["Image file", "Subset", "Medium"]]
    important = important.rename(columns={"Medium": "label"})
    important = important.rename(columns={"Image file": "file_path"})
    important["file_path"] = important["file_path"].apply(lambda x: "./../../../../../DataProcessed/data_256/" + str(x))

    print("Mapping labels...")
    label_mapper = labels.to_dict()[1]
    label_mapper = {v: k for k, v in label_mapper.items()}
    important["label"] = important["label"].map(label_mapper)
    important = important.dropna()
    important["label"].astype(int)
    # number_classes = len(important["label"].drop_duplicates())

    # important = pd.get_dummies(important, columns=["label"], prefix="label_")
    # labels = [x for x in important.columns if "label" in x]

    print("Creating train, val, test dfs...")

    train_df = important[important['Subset'] == 'train']
    val_df = important[important['Subset'] == 'val']
    test_df = important[important['Subset'] == 'test']

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_images = []
    train_labels = []
    # Use a subset of classes to speed up the process. -1 uses all classes.

    for train_image, label in zip(train_df["file_path"], train_df["label"]):
        train_images.append(train_image)
        train_labels.append(label)

    val_images = []
    val_labels = []

    for val_image, label in zip(val_df["file_path"], val_df["label"]):
        val_images.append(val_image)
        val_labels.append(label)

    print('Total train images:', len(train_images), ' with their corresponding', len(train_labels), 'labels')
    # Parameters for the extraction procedure
    batch_size = 20
    input_reshape = (256, 256)
    # Call FNE method on the train set
    fne_features = full_network_embedding(model, train_df, batch_size)
    print('Done extracting features of training set. Embedding size:', fne_features.shape)

    from sklearn import svm

    # Train SVM with the obtained features.
    train_features = fne_features


    print('Done training SVM on extracted features of training set')

    fne_features = full_network_embedding(model, val_images, batch_size)
    print('Done extracting features of val set')

    # Test SVM with the test set.
    val_features = fne_features
    print('Done testing SVM on extracted features of test set')


    with open('train_features.pkl', 'wb') as f:
        pickle.dump(train_features, f)

    with open('train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f)

    with open('val_features.pkl', 'wb') as f:
        pickle.dump(val_features, f)  # Assuming val_features is your validation features

    with open('val_labels.pkl', 'wb') as f:
        pickle.dump(val_labels, f)

    import try_models





