import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn.functional as F
import torch

# OUTPUTS:
# gaze vector (x, y, z) (deep)
# eye heatmaps
# pupil diameter (scalar)

class RegressionHead(nn.Module):
    def __init__(self, in_features, out_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.drop = nn.Dropout(dropout)
        self.pred = nn.Linear(128, out_dim)
        self.log_var = nn.Linear(128, 1)
    def forward(self, x):
        activations = self.drop(F.relu(self.fc1(x)))
        return F.normalize(self.pred(activations), dim=-1), self.log_var(activations)
    
class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Sequential( # depthwise convolution, just spatial mixing
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=2, padding_mode='reflect', groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish()
        )
        self.pw = nn.Sequential( # pointwise convolutions, just channel mixing
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()
        )
    def forward(self, x):
        return self.pw(self.dw(x))
    
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode_conv = DepthwiseSepConv(in_channels, out_channels)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.decode_conv(x)
        return x


class ShinraCNN(nn.Module):
    def __init__(self, backbone, out_channels, heatmap_hw=(112, 112)):
        super().__init__()
        self.backbone = backbone
        self.blocks = []
        self.strided = []
        self.segments = []

        last_skip = 0
        layers = list(backbone.features.children())

        old_conv = layers[0][0] # adjust for single-channel grayscale entry. MobileNetV3 was originally for RGB input
        new_conv = nn.Conv2d(
            in_channels = 1,
            out_channels = old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride = old_conv.stride,
            padding = old_conv.padding,
            bias = old_conv.bias is not None,
        )
        # Average pretrained RGB weights down to single channel
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True)
            )
        layers[0][0] = new_conv

        # Freeze backbone. Undone with succeeding phases
        for param in backbone.parameters():
            param.requires_grad = False
        for module in backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

        for i, layer in enumerate(layers):
            logged = False
            for module in layer.modules():
                if isinstance(module, nn.Conv2d) and not logged:
                    if module.stride != (1, 1):
                        logged = True
                        self.strided.append(layer.out_channels) # for each strided layer of backbone, record output channel count
                        
                        self.segments.append(nn.Sequential(*layers[last_skip:i+1])) # also segment the backbone by skip point, so we can call them sequentially in forward
                        last_skip = i + 1

        self.segments = nn.ModuleList(self.segments)
        self.strided = self.strided[::-1]
        for (o1, o2) in (self.strided[x:x+2] for x in range(len(self.strided)-1)): # for array of [96, 40, 24, 16, 16], make block for [96, 40], [40, 24], so on
            self.blocks.append(DecodeBlock(o1 + o2, o2))
        self.blocks = nn.ModuleList(self.blocks)

        self.heatmap_head = nn.Conv2d(in_channels=self.strided[-1], out_channels=out_channels, kernel_size=1)
        self.diameter_pool = nn.AdaptiveAvgPool2d(1)
        self.diameter_pred = nn.Sequential(nn.Linear(16, 24), nn.BatchNorm1d(24), nn.Linear(24, 1))
        self.diameter_log_var = nn.Linear(16, 1)
        self.gaze_head = RegressionHead(in_features=96, out_dim=3)
        self.gaze_pool = nn.AdaptiveAvgPool2d(1)
        self.heatmap_hw = heatmap_hw
        H, W = heatmap_hw
        ys = torch.linspace(0, 1, H)
        xs = torch.linspace(0, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x.clone())
        self.register_buffer('grid_y', grid_y.clone())
    def forward(self, x, decode=False, border_pad=0):
        if border_pad > 0:
            x = F.pad(x, [border_pad] * 4, mode='reflect')
        skips = []
        # start with a normal pass through the backbone, segment by segment, and saving at skip points
        for i, segment in enumerate(self.segments):
            if i < len(self.segments) - 1:
                x = segment(x)
                skips.append(x)
            else:
                bottleneck = x = segment(x)

        gaze = self.gaze_head(self.gaze_pool(bottleneck).flatten(1))
        skips = skips[::-1] # reverse order of skips, deep -> shallow now.
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
        
        logits = self.heatmap_head(x)
        if border_pad > 0:
            H_tgt, W_tgt = self.heatmap_hw
            ph = (logits.shape[-2] - H_tgt) // 2
            pw = (logits.shape[-1] - W_tgt) // 2
            if ph > 0 or pw > 0:
                logits = logits[:, :, ph:ph + H_tgt, pw:pw + W_tgt]
        if decode:
            B, C, H, W = logits.shape
            weights = torch.softmax(logits.view(B, C, -1), dim=-1).view(B, C, H, W)
            cx = (weights * self.grid_x).sum(dim=(-2, -1))
            cy = (weights * self.grid_y).sum(dim=(-2, -1))
            heatmaps = torch.stack([cx, cy], dim=-1)  # (B, 17, 2)
        else:
            heatmaps = logits
        d_feat = self.diameter_pool(x).flatten(1)
        diameter = self.diameter_pred(d_feat), self.diameter_log_var(d_feat)

        return diameter, heatmaps, gaze
    def thaw(self):
        head_params = (list(self.heatmap_head.parameters()) +
                       list(self.diameter_pred.parameters()) +
                       list(self.diameter_log_var.parameters()) +
                       list(self.gaze_head.parameters()))
        yield None, [{'params': head_params, 'lr': 1e-4}]
        for i, phase in enumerate(self.segments[::-1]):
            for param in phase.parameters():
                param.requires_grad = True
            for module in phase.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()

            heads = (list(self.diameter_pred.parameters()) +
                     list(self.diameter_log_var.parameters()) +
                     list(self.gaze_head.parameters()))
            optim_groups = [{'params': heads, 'lr': 1e-4}, {'params': list(self.heatmap_head.parameters()), 'lr': 2e-4}]
            optim_groups += [{'params': seg.parameters(), 'lr': 2e-4} for seg in self.segments[:i+1]]

            yield phase, optim_groups

