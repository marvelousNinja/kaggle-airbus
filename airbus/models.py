import torch

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            #torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
            #torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        output = self.layers(inputs)
        return output, torch.nn.functional.max_pool2d(output, (2, 2))

class Decoder(torch.nn.Module):
    def __init__(self, shortcut_channels, downsample_channels, out_channels, bilinear=False):
        super().__init__()
        middle_channels = shortcut_channels * 2
        if bilinear:
            self.upscaler = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.upscaler = torch.nn.ConvTranspose2d(downsample_channels, shortcut_channels, (3, 3), stride=2, padding=1, output_padding=1)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(middle_channels, out_channels, (3, 3), padding=1),
            #torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
            #torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, shortcut_input, downsampled_input):
        return self.layers(torch.cat([
            shortcut_input,
            self.upscaler(downsampled_input)
        ], dim=1))

class Unet(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.enc_1 = Encoder(3, 64)
        self.enc_2 = Encoder(64, 128)
        self.enc_3 = Encoder(128, 256)
        self.enc_4 = Encoder(256, 512)
        self.enc_5 = Encoder(512, 1024)
        self.dec_1 = Decoder(512, 1024, 512)
        self.dec_2 = Decoder(256, 512, 256)
        self.dec_3 = Decoder(128, 256, 128)
        self.dec_4 = Decoder(64, 128, 64)
        self.final_conv = torch.nn.Conv2d(64, num_classes, (1, 1))

    def forward(self, inputs):
        x1, x = self.enc_1(inputs)
        x2, x = self.enc_2(x)
        x3, x = self.enc_3(x)
        x4, x = self.enc_4(x)
        x, _ = self.enc_5(x)

        x = self.dec_1(x4, x)
        x = self.dec_2(x3, x)
        x = self.dec_3(x2, x)
        x = self.dec_4(x1, x)
        return self.final_conv(x)

