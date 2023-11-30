import keras
import torchvision.models as models
import torch

#model = models.resnet18(pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)m
model = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3),
    classes=1000,
    classifier_activation="softmax",
)
model.save("./model.h5")