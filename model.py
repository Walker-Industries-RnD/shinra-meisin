import torch
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from input import INPUT_W, INPUT_H
from dann import DomainClassifier

YOLO_MODEL = 'yolo26n.pt'

def fuse(early, mid):
    mid = F.interpolate(mid, size=early.shape[-2:], mode='nearest')
    return torch.cat([early, mid], dim=1)

class RegressionHead(nn.Module): # standard regression, like for scalars and vectors
    def __init__(self, in_features, out_features, pool, log_var=True, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.act = nn.SiLU(inplace=True)
        self.out = nn.Linear(128, out_features)
        self.drop = nn.Dropout(dropout)

        self.log_var = log_var
        self.pool = pool

        if log_var:
            self.log_var = nn.Linear(128, 1)
        if pool:
            self.pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        if self.pool:
            x = self.pool(x).flatten(1)

        pred = self.act(self.fc1(x)) # non-linear activations of hidden layer, [128]
        pred = self.drop(pred)
        if not self.log_var:
            return self.out(pred)
        else:
            return self.out(pred), self.log_var(pred)
        
class HeatmapHead(nn.Module): # heatmaps for precise sub-pixel localization
    def __init__(self, in_channels, out_channels, resolution=(INPUT_H//4, INPUT_W//4), temp=0.1):
        super().__init__()

        # initialize meshgrid, basically giving the weights a coordinate system so that their mult + weighted sum together make expected pos
        ys = torch.linspace(0, 1, int(resolution[0]))
        xs = torch.linspace(0, 1, int(resolution[1]))
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x.clone())
        self.register_buffer('grid_y', grid_y.clone())

        self.temp = temp

        # initialize 1x1 conv gateways. 192 (128+64) channels is a lot
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        logits = self.conv2(self.act(self.conv1(x)))

        B, C, H, W = logits.shape
        soft_map = torch.softmax(logits.view(B, C, -1) / self.temp, dim=-1).view(B, C, H, W) # 2d softmax here - first explode logits to (B, C, 64*48), do softmax, then reshape to (B, C, H, W)
        self.last_soft_map = soft_map  # (B, 17, H, W) — stashed for heatmap visualisation
        pred_x = (soft_map * self.grid_x).sum(dim=[-2, -1])
        pred_y = (soft_map * self.grid_y).sum(dim=[-2, -1])
        pts = torch.stack([pred_x, pred_y], dim=-1)
        return pts

class ShinraCNN(nn.Module):
    def __init__(self):
        super().__init__()
        model = YOLO(YOLO_MODEL) # acquire full YOLO26n model for object detection - but we only need the backbone
        self.backbone = model.model.model[:11]

        layers = list(self.backbone.children())

        # backbone adaptation
        for i, layer in enumerate(layers):
           modules = layer.modules()
           for j, module in enumerate(modules):
               if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)): # part of freezing, so these BNs don't update when frozen
                   module.eval()
               elif isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    if i == 0:
                        self.backbone[0] = Conv(1, module.out_channels, module.kernel_size, module.stride, module.padding) # modify first layer to accept grayscale single-channel input

        # freeze backbone. Undone with succeeding phases
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # backbone tap locations. early/mid retain more spatial context, great for landmarks, deep has global eye context great for gaze & diameter
        early = 2
        mid = 4
        deep = 10

        self.segments = [
            nn.Sequential(*self.backbone[:(early+1)]),
            nn.Sequential(*self.backbone[(early+1):(mid+1)]),
            nn.Sequential(*self.backbone[(mid+1):(deep+1)])
        ]
                   
        self.gaze_head = RegressionHead(in_features=256, out_features=3, pool=True)
        self.diameter_head = RegressionHead(in_features=256, out_features=1, pool=True)
        self.landmark_head = HeatmapHead(in_channels=192, out_channels=17)
        self.dann_clf = DomainClassifier(in_channels=192)
    def forward(self, x, cam_imgs=None):
        def run_backbone(x):
            taps = []
            for segment in self.segments:
                x = segment(x)
                taps.append(x)
            
            return taps
        
        taps = run_backbone(x)

        fused_features = fuse(taps[0], taps[1])

        landmark_pred = self.landmark_head(fused_features) # 64x48x192, fused from 64x48x64 (early) and 32x24x128 (mid) FPN style
        gaze_pred, gaze_lvar = self.gaze_head(taps[2]) # 8x6x256. gaze is a global deal
        diam_pred, diam_lvar = self.diameter_head(taps[2]) # also 8x6x256, we have it pull from a pooled deep tap too because spanning the pupil with respect to the whole eye could use some global context

        if cam_imgs is not None:
            cam_taps = run_backbone(cam_imgs)
            fused_cam_features = fuse(cam_taps[0], cam_taps[1])
            clf_preds = [self.dann_clf(fused_features), self.dann_clf(fused_cam_features)]
        else:
            clf_preds = []

        return landmark_pred, (diam_pred, diam_lvar), (gaze_pred, gaze_lvar), clf_preds
    def thaw(self):
        head_lr_map = torch.linspace(5e-4, 1e-4, len(self.segments))
        segment_lr_map = torch.linspace(2e-4, 2e-5, len(self.segments))

        head_params = (list(self.landmark_head.parameters()) +
                       list(self.diameter_head.parameters()) +
                       list(self.gaze_head.parameters()))
        
        thaw_list = self.segments[::-1]
        
        yield 0, [{'params': head_params, 'lr': 1e-3}]

        for i, segment in enumerate(thaw_list):
            for param in segment.parameters():
                param.requires_grad = True
            for module in segment.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()

            optim_groups = [
                {'params': head_params, 'lr': head_lr_map[i]},
            ]

            if i >= 1:  # phases 2 and 3 — dann_clf becomes active
                optim_groups.append({'params': list(self.dann_clf.parameters()), 'lr': head_lr_map[i]})

            for j, seg in enumerate(thaw_list[:i+1]):
                optim_groups.append({'params': seg.parameters(), 'lr': segment_lr_map[j]})

            yield i+1, optim_groups