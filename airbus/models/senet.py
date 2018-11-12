import torch
from pretrainedmodels import se_resnet50

class Senet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = se_resnet50(pretrained='imagenet')
        self.resnet.last_linear = torch.nn.Linear(self.resnet.last_linear.in_features, num_classes)

    def forward(self, x):
        x = x['image']
        return {'has_ships': self.resnet(x)}
