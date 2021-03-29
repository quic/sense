import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class RealtimeNeuralNet(nn.Module):
    """
    RealtimeNeuralNet is the abstract class for all neural networks used in InferenceEngine.

    Subclasses should overwrite the methods in RealtimeNeuralNet.
    """
    def __init__(self):
        super().__init__()

    def preprocess(self, clip: np.ndarray):
        """
        Pre-process a clip from a video source.
        """
        raise NotImplementedError

    @property
    def step_size(self) -> int:
        """
        Return the step size of the neural network.
        """
        raise NotImplementedError

    @property
    def fps(self) -> int:
        """
        Return the frame per second rate of the neural network.
        """
        raise NotImplementedError

    @property
    def expected_frame_size(self) -> Tuple[int, int]:
        """
        Return the expected frame size of the neural network.
        """
        raise NotImplementedError


class Pipe(RealtimeNeuralNet):

    def __init__(self, feature_extractor, feature_converter):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_converter = feature_converter

    def forward(self, input_tensor):
        feature = self.feature_extractor(input_tensor)
        if isinstance(self.feature_converter, list):
            return [convert(feature) for convert in self.feature_converter]
        return self.feature_converter(feature)

    @property
    def expected_frame_size(self) -> Tuple[int, int]:
        return self.feature_extractor.expected_frame_size

    @property
    def fps(self) -> int:
        return self.feature_extractor.fps

    @property
    def step_size(self) -> int:
        return self.feature_extractor.step_size

    def preprocess(self, clip: np.ndarray):
        return self.feature_extractor.preprocess(clip)


class LogisticRegression(nn.Sequential):

    def __init__(self, num_in, num_out, use_softmax=True, global_average_pooling=True):
        layers = [nn.Linear(num_in, num_out)]
        if use_softmax:
            layers.append(nn.Softmax(dim=-1))
        super(LogisticRegression, self).__init__(*layers)
        self.global_average_pooling = global_average_pooling

    def forward(self, input_tensor):
        if self.global_average_pooling:
            input_tensor = input_tensor.mean(dim=-1).mean(dim=-1)
        return super().forward(input_tensor)


class LogisticRegressionSigmoid(LogisticRegression):

    def __init__(self, **kwargs):
        super().__init__(use_softmax=False, **kwargs)
        self.add_module(str(len(self)), nn.Sigmoid())


class SteppableConv1D(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        self.kernel_size_temporal = kernel_size
        self.stride_temporal = stride
        self.dilation_temporal = dilation
        self.internal_state = None
        self.internal_padding = True

        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         dilation=dilation, **kwargs)

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

        print(x[0, 0, :])
        return super().forward(x)

    def initialize_internal_state(self, x):
        self.internal_state = torch.cat(self.temporal_footprint * [torch.zeros_like(x[:, :, 0:1])],
                                        dim=2)

    def pad_internal_state(self, x):
        x = torch.cat([self.internal_state, x],
                      dim=2)
        self.internal_state = x[:, :, -self.temporal_footprint:]
        return x

    def reset(self):
        self.internal_state = None
        return self

    def train(self, mode=True):
        super().train(mode)
        return self.reset()


class MultiTimestepsLogisticRegression(nn.Sequential):

    def __init__(self, num_in, num_out, kernel, use_softmax=True, global_average_pooling=True):
        layers = [SteppableConv1D(num_in, num_out, kernel)]
        if use_softmax:
            layers.append(nn.Softmax(dim=1))
        super().__init__(*layers)
        self.global_average_pooling = global_average_pooling

    def forward(self, input_tensor):
        if self.global_average_pooling:
            input_tensor = input_tensor.mean(dim=-1).mean(dim=-1)
        input_tensor = input_tensor.unsqueeze(2)
        out = super().forward(input_tensor)
        return out.squeeze(2)
