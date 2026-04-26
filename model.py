import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn.functional as F
import torch

# OUTPUTS:
# gaze vector (x, y, z) (deep)
# eyelid shape (array of 8 2d points) (mid)
# pupil position (x, y) (early)
# pupil diameter (scalar) (early)

# early: 0-3 (224x224x3 -> 28x28x24) CONCAT (14x14x48 UPSAMPLE 28x28x48) = 28x28x72 DEPTHWISE(3x3) POINTWISE(1x1) 28x28x16 
# mid: 4-6 (28x28x24 -> 14x14x40) CONCAT (1x1x576 POINTWISE(1x1) 1x1x40 UPSAMPLE 14x14x40) = 14x14x80 DEPTHWISE(3x3) POINTWISE(1x1) 14x14x48
# deep: 8-13 (14x14x48 -> 1x1x576)

class RegressionHead(nn.Module):
    def __init__(self, in_features, out_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.drop = nn.Dropout(dropout)
        self.pred = nn.Linear(128, out_dim)
        self.log_var = nn.Linear(128, 1)
    def forward(self, x):
        activations = self.drop(F.relu(self.fc1(x)))
        return self.pred(activations), self.log_var(activations)

class ShinraCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(MobileNet_V3_Small_Weights.DEFAULT)
        layers = list(backbone.features.children())

        entry = layers[0][0] # adjust for single-channel grayscale entry. MobileNetV3 was originally for RGB input
        layers[0][0] = nn.Conv2d(
            in_channels = 1,
            out_channels = entry.out_channels,
            kernel_size = entry.kernel_size,
            stride = entry.stride,
            padding = entry.padding,
            bias = entry.bias is not None
        )

        # Freeze backbone. Undone with succeeding phases
        for param in backbone.parameters():
            param.requires_grad = False
        for module in backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

        self.early = nn.Sequential(*layers[0:4])
        self.pool_early = nn.AdaptiveAvgPool2d((2,3))
        self.sep_conv_early = nn.Sequential(
            nn.Conv2d(72, 72, kernel_size=3, padding=1, groups=72, bias=False), # depthwise convolution, only spatial mixing
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 16, kernel_size=1, bias=False), # pointwise convolution, only channel mixing
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pupil = RegressionHead(in_features=96, out_dim=3) # pupil position + diameter

        self.mid = nn.Sequential(*layers[4:7])
        self.pool_mid = nn.AdaptiveAvgPool2d((2,3))
        self.sep_conv_mid = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=3, padding=1, groups=80, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 48, kernel_size=1, bias=False), 
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.eyelid = RegressionHead(in_features=288, out_dim=16) # 8 (x, y) eyelid points

        self.deep = nn.Sequential(*layers[7:])
        self.pool_deep = nn.AdaptiveAvgPool2d((1,1))
        self.pw_conv_deep = nn.Sequential(
            nn.Conv2d(576, 40, kernel_size=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU()
        )
        self.gaze = RegressionHead(in_features=576, out_dim=3) # 3d gaze vector, crown jewel of abstraction
    def forward(self, x):
        early_phase = self.early(x)
        mid_phase = self.mid(early_phase)
        deep_phase = self.deep(mid_phase)

        #early_vec = self.pool_early(early_phase).flatten(1)
        #mid_vec = self.pool_mid(mid_phase).flatten(1)
        deep_vec = self.pool_deep(deep_phase)

        # form U-Net bridges starting with deep->mid
        deep_ups = F.interpolate(self.pw_conv_deep(deep_vec), scale_factor=14, mode='bilinear', align_corners=False)
        deep_mid = torch.cat([mid_phase, deep_ups], dim=1)
        mid_vec = self.sep_conv_mid(deep_mid) # 14x14x48

        # now mid->early
        mid_ups = F.interpolate(mid_vec, scale_factor=2, mode='bilinear', align_corners=False)
        mid_early = torch.cat([early_phase, mid_ups], dim=1)
        early_vec = self.sep_conv_early(mid_early) #  28x28x16

        early_vec = self.pool_early(early_vec).flatten(1)
        mid_vec = self.pool_mid(mid_vec).flatten(1)
        deep_vec = deep_vec.flatten(1)

        pupil_pred = self.pupil(early_vec)
        eyelid_pred = self.eyelid(mid_vec)
        gaze_pred = self.gaze(deep_vec)

        return pupil_pred, eyelid_pred, gaze_pred
    def thaw(self):
        yield None, [{'params': list(self.pupil.parameters()) + list(self.eyelid.parameters()) + list(self.gaze.parameters()), 'lr': 1e-4}]
        phases = (self.deep, self.mid, self.early)
        for i, phase in enumerate(phases):
            for param in phase.parameters():
                param.requires_grad = True
            for module in phase.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()

            heads = list(self.pupil.parameters()) + list(self.eyelid.parameters()) + list(self.gaze.parameters())
            optim_groups = [{'params': heads, 'lr': 1e-4}]
            optim_groups += [{'params': past.parameters(), 'lr': 1e-4} for past in phases[:i]]
            optim_groups += [{'params': phases[i].parameters(), 'lr': 1e-3}]

            yield phase, optim_groups

