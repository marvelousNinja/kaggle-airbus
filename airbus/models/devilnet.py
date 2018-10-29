import torch
import torchvision

class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)

class SCSEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_gate = torch.nn.Sequential(
            GlobalAvgPool2d(),
            torch.nn.Linear(in_channels, in_channels // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction, in_channels),
            torch.nn.Sigmoid()
        )

        self.spatial_gate = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 1, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.spatial_gate(x) * x + self.channel_gate(x).view(x.shape[0], -1, 1, 1) * x

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        )

        self.downsampler = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.relu = torch.nn.ReLU(inplace=True)
        self.scse = SCSEBlock(out_channels)

    def forward(self, x):
        return self.scse(self.relu(self.layers(x) + self.downsampler(x)))


class Devilnet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.center = torch.nn.Sequential(
            Decoder(512, 256),
            torch.nn.MaxPool2d(2)
        )

        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                #self.resnet.conv1,
                torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                self.resnet.bn1,
                self.resnet.relu
                # Maxpool removed intentionally
            ),
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ])

        self.decoders = torch.nn.ModuleList([
            Decoder(256 + 512, 256),
            Decoder(256 + 256, 128),
            Decoder(128 + 128, 64),
            Decoder(64 + 64, 64)
        ])

        self.upscalers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 256, 2, stride=2, groups=256, bias=False),
                torch.nn.Conv2d(256, 256, 1, bias=False)
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 256, 2, stride=2, groups=256, bias=False),
                torch.nn.Conv2d(256, 256, 1, bias=False)
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 128, 2, stride=2, groups=128, bias=False),
                torch.nn.Conv2d(128, 128, 1, bias=False)
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64, 64, 2, stride=2, groups=64, bias=False),
                torch.nn.Conv2d(64, 64, 1,)
            )
        ])

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(512, num_classes, 1)
        )

    def forward(self, x):
        x = self.encoders[0](x)
        x1 = self.encoders[1](x)
        x2 = self.encoders[2](x1)
        x3 = self.encoders[3](x2)
        x4 = self.encoders[4](x3)

        f = self.center(x4)

        d5 = self.decoders[0](torch.cat([self.upscalers[0](f), x4], dim=1))
        d4 = self.decoders[1](torch.cat([self.upscalers[1](d5), x3], dim=1))
        d3 = self.decoders[2](torch.cat([self.upscalers[2](d4), x2], dim=1))
        d2 = self.decoders[3](torch.cat([self.upscalers[3](d3), x1], dim=1))

        f = torch.cat([
            d2,
            torch.nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.functional.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=False),
            torch.nn.functional.interpolate(d5, scale_factor=8, mode='bilinear', align_corners=False)
        ], dim=1)

        return self.classifier(f)
