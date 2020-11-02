import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__reference__ = ["https://github.com/bfortuner/pytorch_tiramisu/blob/master/models/"]
__paper__ = (
    "A Multiple-Feature Reuse Network to Extract Buildings from Remote Sensing Imagery"
)

from torch import Tensor


class _DenseLayer(nn.Sequential):
    """
    Code  borrowed from https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    """

    def __init__(self, num_input_features, growth_rate, drop_rate, batch_momentum):
        super(_DenseLayer, self).__init__()
        self.add_module(
            "norm1", nn.BatchNorm2d(num_input_features, momentum=batch_momentum)
        ),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.Module):
    def __init__(
        self, num_layers, num_input_features, growth_rate, drop_rate, batch_momentum=0.1
    ):
        super(_DenseBlock, self).__init__()
        self.dense_module = nn.ModuleList(
            [
                _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate,
                    drop_rate,
                    batch_momentum=batch_momentum,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        new_feat = list()
        for layer in self.dense_module:
            dense_layer_output = layer(x)
            new_feat.append(dense_layer_output)
            x = torch.cat([x, dense_layer_output], 1)
        return x


class _TransitionDown(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionDown, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class SkipConnectionFilter(nn.Module):
    def __init__(self, input_feature, compression_ratio):
        super().__init__()
        output_feature = int(input_feature * compression_ratio)
        self.filter = nn.Conv2d(
            input_feature, output_feature, kernel_size=1, stride=1, bias=True
        )

    def forward(self, x):
        return self.filter(x)


class _TransitionUp(nn.Module):
    def __init__(self, num_input_features, compression_ratio=0.5):
        super().__init__()
        num_output_features = int(num_input_features * compression_ratio)
        self.Transpose = nn.ConvTranspose2d(
            in_channels=num_input_features,
            out_channels=num_output_features,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x, skip):
        out = self.Transpose(x)
        out = torch.cat([out, skip], 1)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate,
        encoder_block_config,
        decoder_block_config,
        bottleneck_layers,
        num_init_features=48,
        drop_rate=0.0,
        batch_momentum=0.01,
        compression_ratio=0.5,
    ):

        super(DenseNet, self).__init__()
        self.decoder = nn.ModuleList()
        skip_connection_channel_counts = []
        # First convolution

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    ("relu0", nn.ReLU(inplace=True)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(encoder_block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                batch_momentum=batch_momentum,
            )
            self.encoder.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans = _TransitionDown(
                num_input_features=num_features, num_output_features=num_features
            )
            self.encoder.add_module("transition%d" % (i + 1), trans)
            skip_connection_channel_counts.insert(
                0, int(num_features * compression_ratio)
            )

            self.encoder.add_module(
                "skip%d" % (i + 1),
                SkipConnectionFilter(num_features, compression_ratio),
            )

        self.bottle_neck = DenseNetBottleNeck(
            num_features, growth_rate, bottleneck_layers
        )

        num_features = num_features + growth_rate * bottleneck_layers
        for j, num_layers in enumerate(decoder_block_config):
            trans = _TransitionUp(num_features, compression_ratio)
            self.decoder.add_module("decodertransition%d" % (j + 1), trans)

            trans_out_feature = int(num_features * compression_ratio)
            cur_channels_count = trans_out_feature + skip_connection_channel_counts[j]
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=cur_channels_count,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.decoder.add_module("decoderdenseblock%d" % (j + 1), block)
            num_features = cur_channels_count + growth_rate * decoder_block_config[j]

    def forward(self, x):
        encoder_features = self.encoder(x)
        bottle_neck = self.bottle_neck(encoder_features)
        decoder_features = self.decoder(bottle_neck)
        return encoder_features, bottle_neck, decoder_features


class DenseNetBottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.bottle_neck = _DenseBlock(
            growth_rate=growth_rate,
            num_layers=num_layers,
            drop_rate=0.2,
            num_input_features=in_channels,
        )

    def forward(self, x):
        return self.bottle_neck(x)


class MFRN(nn.Module):
    def __init__(
        self,
        classes=1,
        drop_rate=0.2,
        batch_momentum=0.1,
        growth_rate=12,
        layers_per_block=4,
    ):
        super().__init__()
        if isinstance(layers_per_block, int):
            per_block = [layers_per_block] * 11
        elif isinstance(layers_per_block, list):
            assert len(layers_per_block) == 11
            per_block = layers_per_block
        else:
            raise ValueError

        if growth_rate == 12 and layers_per_block == 4:
            final_layer_features = 187
            per_block[0] = 2
            per_block[-1] = 2
        else:
            raise ValueError

        self.mfrn = DenseNet(
            drop_rate=drop_rate,
            batch_momentum=batch_momentum,
            growth_rate=growth_rate,
            encoder_block_config=per_block[0:5],
            decoder_block_config=per_block[6:11],
            bottleneck_layers=per_block[5:6][0],
        )
        self.final_layer = nn.Conv2d(
            final_layer_features, classes, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, input_feature: Tensor) -> Tensor:
        skip_connections = list()
        dense_layer_0 = nn.Sequential(self.mfrn.encoder.conv0, self.mfrn.encoder.relu0)(
            input_feature
        )

        dense_layer_1 = self.mfrn.encoder.denseblock1(dense_layer_0)
        skip_connections.append(self.mfrn.encoder.skip1(dense_layer_1))
        transition_down_1 = self.mfrn.encoder.transition1(dense_layer_1)

        dense_layer_2 = self.mfrn.encoder.denseblock2(transition_down_1)
        skip_connections.append(self.mfrn.encoder.skip2(dense_layer_2))
        transition_down_2 = self.mfrn.encoder.transition2(dense_layer_2)

        dense_layer_3 = self.mfrn.encoder.denseblock3(transition_down_2)
        skip_connections.append(self.mfrn.encoder.skip3(dense_layer_3))
        transition_down_3 = self.mfrn.encoder.transition3(dense_layer_3)

        dense_layer_4 = self.mfrn.encoder.denseblock4(transition_down_3)
        skip_connections.append(self.mfrn.encoder.skip4(dense_layer_4))
        transition_down_4 = self.mfrn.encoder.transition4(dense_layer_4)

        dense_layer_5 = self.mfrn.encoder.denseblock5(transition_down_4)
        skip_connections.append(self.mfrn.encoder.skip5(dense_layer_5))
        transition_down_5 = self.mfrn.encoder.transition5(dense_layer_5)

        bottle_neck = self.mfrn.bottle_neck(transition_down_5)

        transition_up_1 = self.mfrn.decoder.decodertransition1(
            bottle_neck, skip_connections.pop()
        )
        dense_layer_6 = self.mfrn.decoder.decoderdenseblock1(transition_up_1)

        transition_up_2 = self.mfrn.decoder.decodertransition2(
            dense_layer_6, skip_connections.pop()
        )
        dense_layer_7 = self.mfrn.decoder.decoderdenseblock2(transition_up_2)

        transition_up_3 = self.mfrn.decoder.decodertransition3(
            dense_layer_7, skip_connections.pop()
        )
        dense_layer_8 = self.mfrn.decoder.decoderdenseblock3(transition_up_3)

        transition_up_4 = self.mfrn.decoder.decodertransition4(
            dense_layer_8, skip_connections.pop()
        )
        dense_layer_9 = self.mfrn.decoder.decoderdenseblock4(transition_up_4)

        transition_up_5 = self.mfrn.decoder.decodertransition5(
            dense_layer_9, skip_connections.pop()
        )
        dense_layer_10 = self.mfrn.decoder.decoderdenseblock5(transition_up_5)

        final_layer = self.final_layer(dense_layer_10)
        return final_layer
