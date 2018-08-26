import torch

class ConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.layers(inputs)

class DilatedFcn(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = torch.nn.Sequential(
            ConvBnRelu(3, 64, (3, 3), 1),
            ConvBnRelu(64, 64, (3, 3), 1),

            ConvBnRelu(64, 128, (3, 3), 2),
            ConvBnRelu(128, 128, (3, 3), 2),

            ConvBnRelu(128, 256, (3, 3), 3),
            ConvBnRelu(256, 256, (3, 3), 3),
            ConvBnRelu(256, 256, (3, 3), 3),

            ConvBnRelu(256, 256, (3, 3), 3),
            ConvBnRelu(256, 256, (3, 3), 3),
            ConvBnRelu(256, 256, (3, 3), 3),

            ConvBnRelu(256, 256, (3, 3), 2),
            ConvBnRelu(256, 256, (3, 3), 2),

            ConvBnRelu(256, 256, (3, 3), 1),
            ConvBnRelu(256, 256, (3, 3), 1),

            ConvBnRelu(256, 1024, (7, 7), 3),
            ConvBnRelu(1024, 1024, (1, 1), 1),
            torch.nn.Conv2d(1024, num_classes, (1, 1), dilation=1)
        )

    def forward(self, x):
        return self.layers(x)
