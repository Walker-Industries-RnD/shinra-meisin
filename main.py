import torch
from torchvision import models

backbone = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
print(backbone)