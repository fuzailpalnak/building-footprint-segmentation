from torch import nn, Tensor

__reference__ = [
    "Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network",
    "https://arxiv.org/abs/1703.02719",
]


class BoundaryRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)
        self.convolution_layer_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.convolution_layer_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        residual = self.convolution_layer_1(x)
        residual = self.non_linearity(residual)
        residual = self.convolution_layer_2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size=(7, 7)):
        super().__init__()

        # kernel size had better be odd number so as to avoid alignment error
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2

        self.pre_drop = nn.Dropout2d(p=0.1)
        self.convolution_layer_1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)
        )
        self.convolution_layer_2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=(1, kernel_size[1]), padding=(0, pad1)
        )
        self.convolution_layer_3 = nn.Conv2d(
            in_planes, out_planes, kernel_size=(1, kernel_size[1]), padding=(0, pad1)
        )
        self.convolution_layer_4 = nn.Conv2d(
            out_planes, out_planes, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)
        )

    def forward(self, input_feature: Tensor) -> Tensor:
        # x = self.pre_drop(x)
        x_l = self.convolution_layer_1(input_feature)
        x_l = self.convolution_layer_2(x_l)
        x_r = self.convolution_layer_3(input_feature)
        x_r = self.convolution_layer_4(x_r)
        input_feature = x_l + x_r
        return input_feature
