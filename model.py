import torch.nn as nn
import torch
from backbone.select_backbone import select_backbone

class Classifier(nn.Module):
    def __init__(self, net='s3d', dropout=0.5, num_class=400):
        super(Classifier, self).__init__()
        self.backbone, self.param = select_backbone(net)
        feature_size = self.param['feature_size']
        
        self.AvgPool = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        self.Dropout = nn.Dropout3d(dropout)
        self.Conv = nn.Conv3d(feature_size, num_class, kernel_size=1, stride=1, bias=True)

        nn.init.normal_(self.Conv.weight.data, mean=0, std=0.01)
        nn.init.constant_(self.Conv.bias.data, 0.0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.AvgPool(x)
        x = self.Dropout(x)
        x = self.Conv(x)
        return x

if __name__ == '__main__':
    classifier = Classifier()
    data = torch.randn(4,3,64,224,224) # B,T,C,H,W
    logit = classifier(data)
    print(logit.size())

