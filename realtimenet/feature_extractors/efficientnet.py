import torch.nn as nn

from .mobilenet import StridedInflatedMobileNetV2, InvertedResidual, ConvReLU


class StridedInflatedEfficientNet(StridedInflatedMobileNetV2):

    def __init__(self, internal_padding=True):

        super().__init__()

        self.cnn = nn.Sequential(
            ConvReLU(3, 32, 3, stride=2),
            InvertedResidual(32, 24, 3, spatial_stride=1),
            InvertedResidual(24, 32, 3, spatial_stride=2, expand_ratio=6),
            InvertedResidual(32, 32, 3, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             internal_padding=internal_padding),
            InvertedResidual(32, 32, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(32, 32, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(32, 56, 5, spatial_stride=2, expand_ratio=6),
            InvertedResidual(56, 56, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True, temporal_stride=True,
                             internal_padding=internal_padding),
            InvertedResidual(56, 56, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(56, 56, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(56, 112, 3, spatial_stride=2, expand_ratio=6),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             internal_padding=internal_padding),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6, temporal_shift=True, temporal_stride=True,
                             internal_padding=internal_padding),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(112, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             internal_padding=internal_padding),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             internal_padding=internal_padding),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 272, 5, spatial_stride=2, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             internal_padding=internal_padding),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             internal_padding=internal_padding),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 448, 3, spatial_stride=1, expand_ratio=6),
            ConvReLU(448, 1280, 1)
        )
