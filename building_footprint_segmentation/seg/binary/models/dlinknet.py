from torch import nn
from torchvision import models

__author__ = (
    "Zhou, Lichen, Chuang Zhang, and Ming Wu"
    " D-LinkNet: LinkNet With Pretrained Encoder and Dilated Convolution "
    "for High Resolution Satellite Imagery Road Extraction CVPR Workshops. 2018."
)


class DilatedConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_, out, (3, 3), padding=dilation, dilation=(dilation, dilation)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, (1, 1))
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(0, 0),
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, (1, 1))
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class DLinkNet34(nn.Module):
    def __init__(
        self, num_classes=1, res_net_to_use="resnet34", pre_trained_image_net=True
    ):
        super().__init__()
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        res_net = getattr(models, res_net_to_use)(pretrained=pre_trained_image_net)
        self.firstconv = res_net.conv1
        self.firstbn = res_net.bn1
        self.firstrelu = res_net.relu
        self.firstmaxpool = res_net.maxpool
        self.encoder1 = res_net.layer1
        self.encoder2 = res_net.layer2
        self.encoder3 = res_net.layer3
        self.encoder4 = res_net.layer4

        # Center - dilated convolutions
        self.center1 = DilatedConvRelu(512, 512, 1)
        self.center2 = DilatedConvRelu(512, 512, 2)
        self.center3 = DilatedConvRelu(512, 512, 4)
        self.center4 = DilatedConvRelu(512, 512, 8)

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, (3, 3), stride=(2, 2))
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, (3, 3))
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, self.num_classes, (2, 2), padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        c1 = self.center1(e4)
        c2 = self.center2(c1)
        c3 = self.center3(c2)
        c4 = self.center4(c3)

        c = e4 + c1 + c2 + c3 + c4

        # Decoder with Skip Connections
        d4 = self.decoder4(c) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        x_out = f5
        return x_out
