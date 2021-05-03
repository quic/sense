import torch
import torch.nn as nn

from torch.nn.modules.utils import _triple
from sense.downstream_tasks.nn_utils import RealtimeNeuralNet


class SteppableConv3dAs2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        dilation = _triple(dilation)

        self.kernel_size_temporal = kernel_size[0]
        self.stride_temporal = stride[0]
        self.dilation_temporal = dilation[0]
        self.internal_state = None
        self.internal_padding = True

        in_channels *= self.kernel_size_temporal

        super().__init__(in_channels, out_channels,
                         kernel_size[1:], stride=stride[1:], dilation=dilation[1:], **kwargs)

    @property
    def temporal_footprint(self):
        """
        This is used to determine the size of the internal state.
        """
        effective_kernel_size = 1 + (self.kernel_size_temporal - 1) * self.dilation_temporal
        return effective_kernel_size - self.stride_temporal

    def forward(self, x):
        if self.internal_padding:
            if self.internal_state is None:
                self.initialize_internal_state(x)
            x = self.pad_internal_state(x)
        x = self.rearrange_frames(x)
        return super().forward(x)

    def initialize_internal_state(self, x):
        self.internal_state = torch.cat(self.temporal_footprint * [torch.zeros_like(x[0:1])])

    def pad_internal_state(self, x):
        x = torch.cat([self.internal_state, x])
        self.internal_state = x[-self.temporal_footprint:]
        return x

    def rearrange_frames(self, x):
        num_frames = x.shape[0]
        x = torch.cat([x[offset: num_frames - self.kernel_size_temporal + offset + 1]
                       for offset in range(self.kernel_size_temporal)], dim=1)
        x = x[torch.arange(0, x.shape[0], self.stride_temporal)]
        return x

    def reset(self):
        self.internal_state = None
        return self

    def train(self, mode=True):
        super().train(mode)
        return self.reset()


class SteppableSparseConv3dAs2d(SteppableConv3dAs2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        kernel_size = _triple(kernel_size)
        super().__init__(in_channels, out_channels, (1, *kernel_size[1:]),
                         stride=stride, dilation=dilation, **kwargs)
        self.kernel_size_temporal = 3

    def rearrange_frames(self, x):
        # Note: rewrite this to support other kernel sizes (i.e. != 3) and different mixing ratios
        quarter = int(x.shape[1] // 4)
        half = int(x.shape[1] // 2)
        out = torch.zeros_like(x[2:])
        out[:, 0:quarter] = x[0:-2, 0:quarter]
        out[:, quarter:half] = x[1:-1, quarter:half]
        out[:, half:] = x[2:, half:]
        indices = [-1 - offset for offset in range(0, out.shape[0], self.stride_temporal)]
        out = out[indices[::-1]]
        return out


class ConvReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, convlayer=nn.Conv2d,
                 padding=None):
        padding = padding if padding is not None else (kernel_size - 1) // 2
        super(ConvReLU, self).__init__(
            convlayer(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):  # noqa: D101

    def __init__(self, in_planes, out_planes, spatial_kernel_size=3, spatial_stride=1, expand_ratio=1,
                 temporal_shift=False, temporal_stride=False, sparse_temporal_conv=False):
        super().__init__()
        assert spatial_stride in [1, 2]
        hidden_dim = round(in_planes * expand_ratio)
        self.use_residual = spatial_stride == 1 and in_planes == out_planes
        self.temporal_shift = temporal_shift
        self.temporal_stride = temporal_stride

        layers = []
        if expand_ratio != 1:
            # Point-wise expansion
            stride = 1 if not temporal_stride else (2, 1, 1)
            if temporal_shift and sparse_temporal_conv:
                convlayer = SteppableSparseConv3dAs2d
                kernel_size = 1
            elif temporal_shift:
                convlayer = SteppableConv3dAs2d
                kernel_size = (3, 1, 1)
            else:
                convlayer = nn.Conv2d
                kernel_size = 1
            layers.append(ConvReLU(in_planes, hidden_dim, kernel_size=kernel_size, stride=stride,
                                   padding=0, convlayer=convlayer))

        layers.extend([
            # Depth-wise convolution
            ConvReLU(hidden_dim, hidden_dim, kernel_size=spatial_kernel_size, stride=spatial_stride,
                     groups=hidden_dim),
            # Point-wise mapping
            nn.Conv2d(hidden_dim, out_planes, 1, 1, 0),
            # nn.BatchNorm2d(out_planes)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, input_):  # noqa: D102
        output_ = self.conv(input_)
        residual = self.realign(input_, output_)
        if self.use_residual:
            output_ += residual
        return output_

    def realign(self, input_, output_):  # noqa: D102
        n_out = output_.shape[0]
        if self.temporal_stride:
            indices = [-1 - 2 * idx for idx in range(n_out)]
            return input_[indices[::-1]]
        else:
            return input_[-n_out:]


class StridedInflatedMobileNetV2(RealtimeNeuralNet):

    expected_frame_size = (256, 256)
    fps = 16
    step_size = 4
    feature_dim = 1280

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvReLU(3, 32, 3, stride=2),
            InvertedResidual(32, 16),
            InvertedResidual(16, 24, spatial_stride=2, expand_ratio=6),
            InvertedResidual(24, 24, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             sparse_temporal_conv=True),
            InvertedResidual(24, 32, spatial_stride=2, expand_ratio=6),
            InvertedResidual(32, 32, spatial_stride=1, expand_ratio=6, temporal_shift=True, temporal_stride=True,
                             sparse_temporal_conv=True),
            InvertedResidual(32, 32, spatial_stride=1, expand_ratio=6),
            InvertedResidual(32, 64, spatial_stride=2, expand_ratio=6),
            InvertedResidual(64, 64, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             sparse_temporal_conv=True),
            InvertedResidual(64, 64, spatial_stride=1, expand_ratio=6),
            InvertedResidual(64, 64, spatial_stride=1, expand_ratio=6, temporal_shift=True, temporal_stride=True,
                             sparse_temporal_conv=True),
            InvertedResidual(64, 96, spatial_stride=1, expand_ratio=6),
            InvertedResidual(96, 96, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             sparse_temporal_conv=True),
            InvertedResidual(96, 96, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             sparse_temporal_conv=True),
            InvertedResidual(96, 160, spatial_stride=2, expand_ratio=6),
            InvertedResidual(160, 160, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             sparse_temporal_conv=True),
            InvertedResidual(160, 160, spatial_stride=1, expand_ratio=6, temporal_shift=True,
                             sparse_temporal_conv=True),
            InvertedResidual(160, 320, spatial_stride=1, expand_ratio=6),
            ConvReLU(320, self.feature_dim, 1),
        )

    def forward(self, video):
        return self.cnn(video)

    def preprocess(self, clip):
        clip /= 255.
        clip = clip.transpose(0, 1, 4, 2, 3)
        clip = torch.Tensor(clip).float()
        return clip[0]

    @property
    def num_required_frames_per_layer(self):
        """
        Returns a mapping which maps the layer index to the corresponding temporal dependency
        """
        num_required_frames_per_layer = {}
        temporal_dependency = 1
        for index, layer in enumerate(self.cnn[::-1]):
            if isinstance(layer, InvertedResidual):
                if layer.temporal_stride:
                    temporal_dependency = 2 * temporal_dependency - 1
                temporal_dependency = temporal_dependency + int(layer.temporal_shift * 2)
            num_required_frames_per_layer[len(self.cnn) - 1 - index] = temporal_dependency
            num_required_frames_per_layer[-1 - index] = temporal_dependency
        num_required_frames_per_layer[0] = temporal_dependency
        return num_required_frames_per_layer

    @property
    def num_required_frames_per_layer_padding(self):
        """
        Returns a mapping which maps the layer index to the minimum number of input frame
        """
        num_required_frames_per_layer = {}
        temporal_dependency = 1

        for index, layer in enumerate(self.cnn[::-1]):
            if isinstance(layer, InvertedResidual):
                if layer.temporal_stride:
                    temporal_dependency = 2 * temporal_dependency

            num_required_frames_per_layer[len(self.cnn) - 1 - index] = temporal_dependency
            num_required_frames_per_layer[-1 - index] = temporal_dependency

        num_required_frames_per_layer[0] = temporal_dependency

        return num_required_frames_per_layer
