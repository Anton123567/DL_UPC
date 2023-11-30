import numpy as np
import cv2
import pandas as pd
import torch

import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn

import os
import tqdm
from tqdm.auto import tqdm

from cdataset import CustomDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


import ssl
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

from cdataset import CustomDataset
import torchvision.models as models
import torchvision.models as models

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':

    # HYPERPARAMETERS
    FREEZE_ALL = False
    FC_SINGLE = False
    BATCH_SIZE = 64
    NUM_WORKERS = os.cpu_count()
    LR = 0.0001
    # Set number of epochs
    NUM_EPOCHS = 150

    # DATASET DF SETUP
    dataset = pd.read_csv("../../DataMeta/MAMe_dataset.csv")
    labels = pd.read_csv("../../DataMeta/MAMe_labels.csv", header=None)
    toy_data = pd.read_csv("../../DataMeta/MAMe_toy_dataset.csv")

    important = dataset[["Image file", "Subset", "Medium"]]
    important = important.rename(columns={"Medium": "label"})
    important = important.rename(columns={"Image file": "file_path"})
    important["file_path"] = important["file_path"].apply(lambda x: "../../DataProcessed/data_256/" + str(x))

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

    print(f"Train df of length {len(train_df)}, validation df of length {len(val_df)} and test df of length {len(test_df)} is created.")
    print(f"Creating dataset and dataLoader with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

    augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Random crop with resizing to 64x64
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Random color jitter
        ]

    train_transform = transforms.Compose(augmentations + [transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    train_dataset = CustomDataset(train_df, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    val_transform = torchvision.transforms.Compose([
        transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = CustomDataset(val_df, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device is {device}.')


    # train step is for ONE epoch

    def train_step(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler,
                   accuracy_fn,
                   device: torch.device = device):

        train_loss, train_acc = 0, 0

        # put model to training mode
        model.train()

        # Add progress bar also for batches
        tk0 = tqdm(data_loader, total=int(len(data_loader)))

        # Loop through the batches
        for batch, (X, y) in enumerate(tk0):  # for len(train_dataloder) number of batches
            # for each batch take X ([32,1,28,28]) and y ([32])
            # given shapes are as it is in our example
            # Put the data to target device
            y = y.long()
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)  # gives logits

            # 2. Calculate the loss and accuracy PER batch
            loss = loss_fn(y_pred, y)  # crossentrophy works with logits
            train_loss += loss  # accumulate train loss (add the loss of the whole batch)
            train_acc += accuracy_fn(y_true=y,
                                     y_pred=y_pred.argmax(dim=1))  # from logits -> prediction labels

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Update model's parameters with optimizer - once *per batch*
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Print what's happening
            # if batch % 400 == 0:
            #  print(f"Looked at {batch * len(X)} / {len(data_loader.dataset)} samples.")
            # tk0.set_description('Train Epoch: [{}/{}] Loss: {:.4f} '
            #                       'Train Sup Acc: {:.4f}'.format(epoch, epochs, total_loss / total_num, acc))

        # train_loss after 1 epoch: sum of train losses for each batch / number of batches
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        # print(f"Train loss: {train_loss: .5f} | Train acc: {train_acc: .2f} %")

        return train_loss, train_acc

        # define validation step


    def val_step(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 accuracy_fn,
                 device: torch.device = device):
        val_loss, val_acc = 0, 0

        # Put model in eval mode
        model.eval()
        # Turn on inference mode
        with torch.no_grad():
            for X, y in data_loader:
                # Send data to target device
                y = y.long()
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                val_pred = model(X)  # raw logits

                # 2. Calculate the loss
                val_loss += loss_fn(val_pred, y)

                val_acc += accuracy_fn(y_true=y,
                                       y_pred=val_pred.argmax(dim=1))  # logits -> prediction labels

        val_loss /= len(data_loader)
        val_acc /= len(data_loader)

        print(f"Validation loss: {val_loss: .5f} | Validation acc: {val_acc: .2f} % \n")

        return val_loss, val_acc

        # IN ORDER TO CHECK INPUT-OUTPUT SIZES
        # print("Shape of the input without unsqueeze:", train_features[0].shape)
        # model.eval()
        # with torch.inference_mode():
        #     model(train_features[0].unsqueeze(dim=0).to(device))


    # Calculate accuracy (a classification metric)
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


    # PRE-TRAINED MODEL
    # model = models.resnet18(pretrained=True) # internet required
    model = torch.load('../pretrained_resnet50.pth')
    model = model.to(device)

    # vgg
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    # delete layer4
    model.layer4 = Identity().to(device)

    # REPLACE THE CLASSIFICATION LAYER
    if FC_SINGLE:
        model.fc = torch.nn.Linear(in_features=1024, out_features=29, bias=True).to(device)
        print('Smaller FC: only one layer with 2048 -> 29')
    else:
        model.fc = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(in_features=1024,
                                  out_features=128),
                        torch.nn.ReLU(),
                        nn.Dropout2d(p=0.5),
                        nn.Linear(in_features=128,
                                  out_features=29)).to(device)
        print('Bigger FC layers, replaced. 2 Linear 2048 -> 128 -> 29, dropout of 0.5')

    # # REPLACE LAYER 4
    # model_nottrained = torch.load('../NOTtrained_resnet18.pth')
    # model_nottrained = model_nottrained.to(device)
    # model.layer4 = model_nottrained.layer4


    # FREEZE SOME LAYERS
    if FREEZE_ALL:
        for param in model.parameters():  # freeze all
            param.requires_grad = False
        for param in model.fc.parameters():  # unfreeze fc
            param.requires_grad = True
        print('All layers are frozen other than the FC.')

    else:
        # FREEZE SOME LAYERS
        for param in model.parameters(): # freeze all at first, in the end beginning, layer 1 and 2 frozen
            param.requires_grad = False

        # for param in model.layer3.parameters():  # unfreeze layer 3
        #     param.requires_grad = True

        for param in model.layer3.parameters():  # unfreeze layer 3
            param.requires_grad = True

        for param in model.fc.parameters(): # unfreeze fc -> train from scratch
            param.requires_grad = True


    num_train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable params: {num_train_param}')

    # from torchinfo import summary
    # input_size=(32, 3, 256, 256)
    # summary(model,  col_names=["input_size","output_size", "num_params", "trainable"], input_size= input_size)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    TOTAL_STEPS = NUM_EPOCHS * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=TOTAL_STEPS,  # Maximum number of iterations.
                                  eta_min=0.0000001)  # Minimum learning rate.


    results = {'epoch': [], 'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    best_loss = 100.0
    best_acc = 0.0

    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"Epoch: {epoch} \n --------")
        training_loss, training_acc = train_step(model=model,
                                                 data_loader=train_loader,
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 accuracy_fn=accuracy_fn,
                                                 device=device)

        valid_loss, valid_acc = val_step(model=model,
                                         data_loader=val_loader,
                                         loss_fn=loss_fn,
                                         accuracy_fn=accuracy_fn,
                                         device=device)

        results['epoch'].append(epoch)
        results['train_loss'].append(training_loss.cpu().detach().numpy())
        results['train_acc'].append(training_acc)

        results['val_loss'].append(valid_loss.cpu().detach().numpy())
        results['val_acc'].append(valid_acc)

        data_frame = pd.DataFrame(data=results)
        data_frame = data_frame.set_index('epoch')
        data_frame.to_csv('./results/results.csv')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './results/model_cpt.pth')
            torch.save(model, './model_bestloss.pth')

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },  './results/model_cpt_bestAcc.pth')
            torch.save(model, './model_bestAcc.pth')


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

loss_fn = nn.CrossEntropyLoss()

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

