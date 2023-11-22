import torchvision.models as models
import torch

model = models.resnet18(pretrained=True)
torch.save(model, "model_resnet18")