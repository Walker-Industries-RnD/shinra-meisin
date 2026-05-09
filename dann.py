# DANN - Domain Adversarial Neural Network
from torch.autograd import Function
import torch
from torch import nn

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
class DomainClassifier(nn.Module):
    def __init__(self, in_channels=192):
        super().__init__()
        self.grl = GradientReversal(lambda_=1.0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.grl(x)
        x = self.pool(x)
        return self.classifier(x)