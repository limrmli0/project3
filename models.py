import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class R34_ver1(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver1, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(OrderedDict([*(list(resnet.named_children())[:-2])])) # prepare backbone without final fc layer

        # freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False 
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # customized classifier layers
        self.classfier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classfier(feat_1d)

        return logit
