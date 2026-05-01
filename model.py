import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn.functional as F
import torch

# OUTPUTS:
# gaze vector (x, y, z) (deep)
# eye heatmaps
# pupil diameter (scalar)

LR_MAX = 1e-4 # maximum LR for the segments
LR_MIN = 1e-5 # minimum LR

HEATMAP_BORDER = 8 # pixels masked from each side, both heatmap loss (0.25x) and upsample results

class RegressionHead(nn.Module):
    def __init__(self, in_features, out_dim, dropout=0.3, normalize=False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.drop = nn.Dropout(dropout)
        self.pred = nn.Linear(128, out_dim)
        self.log_var = nn.Linear(128, 1)
        self.normalize = normalize
    def forward(self, x):
        activations = self.drop(F.relu(self.fc1(x)))
        pred = self.pred(activations)
        if self.normalize:
            pred = F.normalize(pred, dim=-1)
        return pred, self.log_var(activations)
    
class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Sequential( # depthwise convolution, just spatial mixing
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, padding_mode='zeros', groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.pw = nn.Sequential( # pointwise convolutions, just channel mixing
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.pw(self.dw(x))
    
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode_conv = DepthwiseSepConv(in_channels, out_channels)
    def forward(self, x, skip):
        b = max(1, round(HEATMAP_BORDER * 0.25))  # scaled-down border mask before upsample (full HEATMAP_BORDER is too steep at deep resolutions)
        x = x.clone()
        x[..., :b, :] = 0;  x[..., -b:, :] = 0
        x[..., :, :b] = 0;  x[..., :, -b:] = 0
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear')
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
        
        self.diameter_pool = nn.ModuleList([nn.AdaptiveAvgPool2d((2,2)), nn.AdaptiveAvgPool2d(1)])
        self.diameter_head = RegressionHead(in_features=80, out_dim=1, normalize=False)
        
        self.gaze_head = RegressionHead(in_features=96, out_dim=3, normalize=True)
        self.gaze_pool = nn.AdaptiveAvgPool2d(1)
        self.heatmap_hw = heatmap_hw
        H, W = heatmap_hw
        ys = torch.linspace(0, 1, H)
        xs = torch.linspace(0, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x.clone())
        self.register_buffer('grid_y', grid_y.clone())
    def forward(self, x, decode=False, return_decoder_feats=False):  # [decoder-debug]
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
        decoder_feats = []  # [decoder-debug]
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
            if return_decoder_feats:  # [decoder-debug]
                decoder_feats.append(x.detach().cpu())
        
        logits = self.heatmap_head(x)
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

        dia_spp = [dia_pool(x).flatten(1) for dia_pool in self.diameter_pool] # 112x112x16 -> 2x2x16 + 1x1x16 -> flattened to [80]
        d_feat = torch.cat(dia_spp, dim=1)

        diameter = self.diameter_head(d_feat)

        if return_decoder_feats:  # [decoder-debug]
            return diameter, heatmaps, gaze, decoder_feats
        return diameter, heatmaps, gaze
    def thaw(self):
    
        head_lr_map = torch.linspace(5e-4, 2e-4, len(self.segments))
        segment_lr_map = torch.linspace(2e-4, 4e-5, len(self.segments))

        head_params = (list(self.heatmap_head.parameters()) +
                       list(self.diameter_head.parameters()) +
                       list(self.gaze_head.parameters()))
        
        yield None, [{'params': head_params, 'lr': 1e-3}]

        for i, phase in enumerate(self.segments[::-1]):
            for param in phase.parameters():
                param.requires_grad = True
            for module in phase.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()

            optim_groups = [{'params': head_params, 'lr': head_lr_map[i]}]
            
            for j, seg in enumerate(self.segments[:i+1]):
                optim_groups.append({'params': seg.parameters(), 'lr': segment_lr_map[j]})

            yield phase, optim_groups

