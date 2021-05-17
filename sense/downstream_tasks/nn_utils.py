import numpy as np
import torch.nn as nn
from typing import Tuple


class RealtimeNeuralNet(nn.Module):
    """
    RealtimeNeuralNet is the abstract class for all neural networks used in InferenceEngine.

    Subclasses should overwrite the preprocess method.
    """
    def __init__(self, step_size: int, fps: int, expected_frame_size: Tuple[int, int]):
        """
        :param step_size:
            The temporal step size of the neural network, i.e. how many frames should be consumed before outputting
            the next prediction.
        :param fps:
            The frame per second rate of the neural network.
        :param expected_frame_size:
            The expected frame size of the neural network.
        """
        super().__init__()
        self.step_size = step_size
        self.fps = fps
        self.expected_frame_size = expected_frame_size

    def preprocess(self, clip: np.ndarray):
        """
        Pre-process a clip from a video source.
        """
        raise NotImplementedError


class Pipe(RealtimeNeuralNet):

    def __init__(self, feature_extractor, feature_converter):
        super().__init__(step_size=feature_extractor.step_size,
                         fps=feature_extractor.fps,
                         expected_frame_size=self.feature_extractor.expected_frame_size)
        self.feature_extractor = feature_extractor
        self.feature_converter = feature_converter

    def forward(self, input_tensor):
        feature = self.feature_extractor(input_tensor)
        if isinstance(self.feature_converter, list):
            return [convert(feature) for convert in self.feature_converter]
        return self.feature_converter(feature)

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
