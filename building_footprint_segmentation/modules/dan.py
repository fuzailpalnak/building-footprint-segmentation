from torch import nn, Tensor

__reference__ = [
    "Building Extraction in Very High Resolution Imagery by Dense-Attention Networks",
    "https://www.mdpi.com/2072-4292/10/11/1768/htm",
]


class SpatialAttentionFusionModule(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(low_level_features: Tensor, high_level_features: Tensor) -> Tensor:
        """

        :param low_level_features: features extracted from backbone
        :param high_level_features: up sampled features
        :return:
        """
        high_level_features_sigmoid = high_level_features.sigmoid()
        weighted_low_level_features = high_level_features_sigmoid * low_level_features

        feature_fusion = weighted_low_level_features + high_level_features
        return feature_fusion
