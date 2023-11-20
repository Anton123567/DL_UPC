import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/anton/Documents/Uni/Deep_Learning'])


import os
import zipfile

import matplotlib
import seaborn as sn
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from DL_UPC.Netclasses import RNN
from DL_UPC.cdataset import CustomDataset



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device is {device}.')

LABELMAP = {}


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def get_val_loader():
    # (HYPER)PARAMETERS
    BATCH_SIZE = 32 #changed from 8!
    NUM_WORKERS = os.cpu_count()
    LR = 0.0001

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

    test_df = important[important['Subset'] == 'test']

    test_df = test_df.reset_index(drop=True)

    test_transform = torchvision.transforms.Compose([
                                   transforms.Resize((256, 256)),
                                   torchvision.transforms.ToTensor()])
    val_dataset = CustomDataset(test_df, test_transform)
    val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    return val_loader



# Assuming `MyModel` is the class name of your model.
model = RNN(3,22,29)
model.to(device)

LR = 0.0001
# Assuming `optimizer` was originally created as below (adjust as necessary for your specific optimizer and parameters)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Load the checkpoint
checkpoint = torch.load('./models/model_cpt.pth')

# Restore the model and optimizer state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss_fn = nn.CrossEntropyLoss()

# If you're loading the model for inference only, switch to eval mode
model.eval()

model.eval()
test_loss = 0.0
test_acc = 0
test_loader = get_val_loader()

all_preds = []
all_trueths = []

with torch.no_grad():
    for i, (X, y) in enumerate(tqdm(test_loader)):
        all_trueths.append(y.numpy().tolist())
        # Send data to target device
        y = y.long()
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        test_pred = model(X)  # raw logits

        all_preds.append((torch.max(torch.exp(test_pred), 1)[1]).data.cpu().numpy().tolist())

        # 2. Calculate the loss
        test_loss += loss_fn(test_pred, y)

        test_acc += accuracy_fn(y_true=y,
                               y_pred=test_pred.argmax(dim=1))  # logits -> prediction labels

test_loss /= len(test_loader)
test_acc /= len(test_loader)

print(f"Test loss: {test_loss: .5f} | Test acc: {test_acc: .2f} % \n")

labels = pd.read_csv("../DataMeta/MAMe_labels.csv", header=None)

classes = labels[1]

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list



cf_matrix = confusion_matrix(flatten_extend(all_trueths), flatten_extend(all_preds))

new_shape = (cf_matrix.shape[0] + 1, cf_matrix.shape[1] + 1)
new_matrix = np.zeros(new_shape, dtype=cf_matrix.dtype)

new_matrix[:-1, :-1] = cf_matrix

df_cm = pd.DataFrame(new_matrix / np.sum(new_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
df_cm.fillna(0, inplace=True)
import matplotlib
matplotlib.use('Qt5Agg')
#sn.set(font_scale=1.4)
plt.figure(figsize = (20,15))
sn.heatmap(df_cm, annot=False)
plt.tight_layout()
plt.savefig('output.png')
plt.show()







