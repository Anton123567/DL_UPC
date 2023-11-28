import torchvision.models as models
import torch

model = models.resnet18(pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
torch.save(model, "densenet201")