from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import models

__paper__ = "https://arxiv.org/pdf/1611.06612.pdf"
__reference__ = "https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/refine_net.py"

"""

backbone_features - represent the output of backbone network 
refine_block_features - represent the output of Refine Block

"""


def convolution_3x3(in_planes, out_planes, stride=1, padding=1, bias=True):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias
    )


def convolution_1x1(in_planes, out_planes, stride=1, padding=0, bias=True):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )


class ResidualConvolutionUnit(nn.Module):
    """
    Section 3.2:
        The first part of each RefineNet block consists of an adaptive convolution set that mainly fine tunes
        the pre trained the ResNet weights
    """

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)

        self.convolution_layer_1 = convolution_3x3(in_planes, out_planes)
        self.convolution_layer_2 = convolution_3x3(out_planes, out_planes)

    def forward(self, x):
        residual = x

        x = self.non_linearity(x)
        x = self.convolution_layer_1(x)

        x = self.non_linearity(x)
        x = self.convolution_layer_2(x)

        x = residual + x

        return x


class MultiResolutionFusion(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.convolution_layer_lower_inputs = convolution_3x3(in_planes, out_planes)
        self.convolution_layer_higher_inputs = convolution_3x3(out_planes, out_planes)

    def forward(self, backbone_features, refine_block_features=None):
        if refine_block_features is None:
            """
            Suggests RefineNet-4
            """
            return self.convolution_layer_higher_inputs(backbone_features)
        else:
            backbone_features = self.convolution_layer_higher_inputs(backbone_features)

            refine_block_features = self.convolution_layer_lower_inputs(
                refine_block_features
            )
            refine_block_features = F.interpolate(
                refine_block_features,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )

            return refine_block_features + backbone_features


class ChainedResidualPooling(nn.Module):
    """
    Section-1:
        Chained residual pooling is able to capture background context from a large image region
    """

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)
        self.convolution_layer_1 = convolution_3x3(
            in_planes=in_planes, out_planes=out_planes
        )
        self.max_pooling_layer = nn.MaxPool2d((5, 5), stride=1, padding=2)

    def forward(self, x):
        x_non_linearity = self.non_linearity(x)
        first_pass = self.max_pooling_layer(x_non_linearity)
        first_pass = self.convolution_layer_1(first_pass)

        intermediate_sum = first_pass + x_non_linearity

        second_pass = self.max_pooling_layer(first_pass)
        second_pass = self.convolution_layer_1(second_pass)

        x = second_pass + intermediate_sum
        return x


class RefineBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.residual_convolution_unit = ResidualConvolutionUnit(out_planes, out_planes)
        self.multi_resolution_fusion = MultiResolutionFusion(in_planes, out_planes)
        self.chained_residual_pooling = ChainedResidualPooling(out_planes, out_planes)

    def forward(self, backbone_features, refine_block_features=None):
        """

        :param backbone_features: input from backbone network
        :param refine_block_features: input from refine net block
        :return:
        """
        if refine_block_features is None:
            """
            Suggests RefineNet-4
            """
            """
            Section 3.2:
                The first part of each RefineNet block consists of an adaptive convolution set that mainly fine tunes
                the pre trained the ResNet weights 
            """
            x = self.residual_convolution_unit(backbone_features)
            x = self.residual_convolution_unit(x)

            """
            section 3.2 - 
                Multi-resolution fusion :   
                    If there is only one input path (e.g , the case of RefineNet-4 the input will directly go thorough
            """
            x = self.multi_resolution_fusion(x)
            x = self.chained_residual_pooling(x)

            x = self.residual_convolution_unit(x)

            return x
        else:
            """
            Section 3.2:
                The first part of each RefineNet block consists of an adaptive convolution set that mainly fine tunes
                the pre trained the ResNet weights
            """
            x = self.residual_convolution_unit(backbone_features)
            x = self.residual_convolution_unit(x)

            x = self.multi_resolution_fusion(x, refine_block_features)
            x = self.chained_residual_pooling(x)

            x = self.residual_convolution_unit(x)

            return x


class ReFineNet(nn.Module):
    def __init__(
        self,
        res_net_to_use="resnet34",
        pre_trained_image_net=True,
        top_layers_trainable=True,
        num_classes=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        res_net = getattr(models, res_net_to_use)(pretrained=pre_trained_image_net)

        if not top_layers_trainable:
            for param in res_net.parameters():
                param.requires_grad = False

        self.layer0 = nn.Sequential(
            res_net.conv1, res_net.bn1, res_net.relu, res_net.maxpool
        )
        self.layer1 = res_net.layer1
        self.layer2 = res_net.layer2
        self.layer3 = res_net.layer3
        self.layer4 = res_net.layer4
        if res_net_to_use == "resnet50":
            layers_features = [256, 512, 1024, 2048]

        elif res_net_to_use == "resnet34":
            layers_features = [64, 128, 256, 512]

        else:
            raise NotImplementedError

        """
        section 3.1 -
            In practice each ResNet output is passed through one convolution layer to adapt the dimensionality
        """
        self.convolution_layer_4_dim_reduction = convolution_3x3(
            in_planes=layers_features[-1], out_planes=512
        )
        self.convolution_layer_3_dim_reduction = convolution_3x3(
            in_planes=layers_features[-2], out_planes=256
        )
        self.convolution_layer_2_dim_reduction = convolution_3x3(
            in_planes=layers_features[-3], out_planes=256
        )
        self.convolution_layer_1_dim_reduction = convolution_3x3(
            in_planes=layers_features[-4], out_planes=256
        )

        self.refine_block_4 = RefineBlock(
            in_planes=512,
            out_planes=512,
        )
        self.refine_block_3 = RefineBlock(in_planes=512, out_planes=256)
        self.refine_block_2 = RefineBlock(
            in_planes=256,
            out_planes=256,
        )
        self.refine_block_1 = RefineBlock(
            in_planes=256,
            out_planes=256,
        )

        """
        Section 3.2:
            The final step of Each RefineNet block is another residual convolution unit .
            This results in a sequence of three RCU between each block.
            To reflect this behaviour in the last RefineNet-1 block, we place two additional RCU
        """
        self.residual_convolution_unit = ResidualConvolutionUnit(
            in_planes=256, out_planes=256
        )

        self.final_layer = convolution_1x1(in_planes=256, out_planes=self.num_classes)

    def forward(self, input_feature: Tensor) -> Tensor:

        layer_0_output = self.layer0(input_feature)
        layer_1_output = self.layer1(layer_0_output)  # 1/4
        layer_2_output = self.layer2(layer_1_output)  # 1/8
        layer_3_output = self.layer3(layer_2_output)  # 1/16
        layer_4_output = self.layer4(layer_3_output)  # 1/32

        backbone_layer_4 = self.convolution_layer_4_dim_reduction(
            layer_4_output
        )  # 1/32
        backbone_layer_3 = self.convolution_layer_3_dim_reduction(
            layer_3_output
        )  # 1/16
        backbone_layer_2 = self.convolution_layer_2_dim_reduction(layer_2_output)  # 1/8
        backbone_layer_1 = self.convolution_layer_1_dim_reduction(layer_1_output)  # 1/4

        refine_block_4 = self.refine_block_4(backbone_layer_4)
        refine_block_3 = self.refine_block_3(backbone_layer_3, refine_block_4)
        refine_block_3 = self.refine_block_2(backbone_layer_2, refine_block_3)
        refine_block_1 = self.refine_block_1(backbone_layer_1, refine_block_3)

        """
        Section 3.2:
            The final step of Each RefineNet block is another residual convolution unit .
            This results in a sequence of three RCU between each block.
            To reflect this behaviour in the last RefineNet-1 block, we place two additional RCU
        """
        residual_convolution_unit = self.residual_convolution_unit(refine_block_1)
        residual_convolution_unit = self.residual_convolution_unit(
            residual_convolution_unit
        )

        final_map = self.final_layer(residual_convolution_unit)
        final_map = F.interpolate(
            final_map, scale_factor=4, mode="bilinear", align_corners=True
        )
        return final_map
