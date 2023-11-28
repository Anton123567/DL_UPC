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

#from cdataset import CustomDataset

import ssl
import torchvision.models as models

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':

    # PRE-TRAINED MODEL
    # model = models.resnet18(pretrained=True)
    # torch.save(model, './pretrained_resnet18.pth')

    # model = models.resnet18(pretrained=False)
    # torch.save(model, './NOTtrained_resnet18.pth')
    #
    # model = models.resnet50(pretrained=True)
    # torch.save(model, './pretrained_resnet50.pth')
    #
    # model = models.resnet18(pretrained=False)
    # torch.save(model, './NOTtrained_resnet50.pth')
    #
    # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    # torch.save(model, './pretrained_efficientnet_b0.pth')

    # model = models.efficientnet_v2_s(pretrained = True)
    # torch.save(model, './pretrained_efficientnet_b0.pth')

    model = models.vgg16(pretrained = True)
    torch.save(model, './pretrained_vgg16.pth')



