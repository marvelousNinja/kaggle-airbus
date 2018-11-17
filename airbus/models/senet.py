import torch
import pretrainedmodels

class Senet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = pretrainedmodels.se_resnet50(pretrained='imagenet')
        self.resnet.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.resnet.last_linear = torch.nn.Linear(self.resnet.last_linear.in_features, num_classes)

    def forward(self, x):
        x = x['image']
        return {'has_ships': self.resnet(x)}
