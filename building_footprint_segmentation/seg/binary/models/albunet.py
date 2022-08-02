from torch import nn
import torch
from torchvision import models

__author__ = (
    "Shvets, Alexey & Iglovikov, Vladimir & Rakhlin, Alexander & Kalinin, Alexandr. (2018). Angiodysplasia"
    " Detection and Localization Using Deep Convolutional Neural Networks. "
)

__code__ = "https://github.com/ternaus/TernausNet/blob/master/unet_models.py"


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, (3, 3), padding=1)


class ConvolutionReLu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvolutionReLu, self).__init__()
        self.convolution = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convolution(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(
        self, in_channels, middle_channels, out_channels, is_deconvolution=True
    ):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconvolution:
            self.block = nn.Sequential(
                ConvolutionReLu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels,
                    out_channels,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                ConvolutionReLu(in_channels, middle_channels),
                ConvolutionReLu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class AlBuNet(nn.Module):
    """
    UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) ict_net
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    https://github.com/ternaus/robot-surgery-segmentation
    """

    def __init__(
        self,
        num_classes=1,
        num_filters=32,
        pre_trained=False,
        is_deconvolution=False,
        res_net_to_use="resnet34",
    ):
        """
        :param num_classes:
        :param num_filters:
        :param pre_trained:
            False - no pre-trained network is used
            True  - ict_net is pre-trained with resnet34
        :is_deconvolution:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = getattr(models, res_net_to_use)(pretrained=pre_trained)

        # self.ict_net = torchvision.models.resnet34(pretrained=pre_trained)

        if res_net_to_use == "resnet50":
            layers_features = [256, 512, 1024, 2048]

        elif res_net_to_use == "resnet34":
            layers_features = [64, 128, 256, 512]

        else:
            raise NotImplementedError

        self.non_linearity = nn.ReLU(inplace=True)

        self.convolution_1 = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool
        )

        self.convolution_2 = self.encoder.layer1

        self.convolution_3 = self.encoder.layer2

        self.convolution_4 = self.encoder.layer3

        self.convolution_5 = self.encoder.layer4

        self.center = DecoderBlock(
            layers_features[-1], num_filters * 8 * 2, num_filters * 8, is_deconvolution
        )

        self.dec5 = DecoderBlock(
            layers_features[-1] + num_filters * 8,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconvolution,
        )
        self.dec4 = DecoderBlock(
            layers_features[-2] + num_filters * 8,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconvolution,
        )
        self.dec3 = DecoderBlock(
            layers_features[-3] + num_filters * 8,
            num_filters * 4 * 2,
            num_filters * 2,
            is_deconvolution,
        )
        self.dec2 = DecoderBlock(
            layers_features[-4] + num_filters * 2,
            num_filters * 2 * 2,
            num_filters * 2 * 2,
            is_deconvolution,
        )
        self.dec1 = DecoderBlock(
            num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconvolution
        )
        self.dec0 = ConvolutionReLu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        convolution_1 = self.convolution_1(x)
        convolution_2 = self.convolution_2(convolution_1)
        convolution_3 = self.convolution_3(convolution_2)
        convolution_4 = self.convolution_4(convolution_3)
        convolution_5 = self.convolution_5(convolution_4)

        center = self.center(self.pool(convolution_5))

        dec5 = self.dec5(torch.cat([center, convolution_5], 1))

        dec4 = self.dec4(torch.cat([dec5, convolution_4], 1))
        dec3 = self.dec3(torch.cat([dec4, convolution_3], 1))
        dec2 = self.dec2(torch.cat([dec3, convolution_2], 1))

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)

        return x_out
