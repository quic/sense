import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from sense import RESOURCES_DIR


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

    def load_weights_from_resources(self, checkpoint_path: str, strict: bool = True):
        """
        Load weights from provided checkpoint file, unless the TRAVIS environment
        variable is defined.
        """
        if not os.getenv('TRAVIS', False) == 'true':
            checkpoint = load_weights_from_resources(checkpoint_path)
            self.load_state_dict(checkpoint, strict=strict)
        else:
            print('Weights are not loaded on Travis.')


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


def load_weights_from_resources(checkpoint_path: str):
    """
    Load weights from a checkpoint file located in the resources folder.

    :param checkpoint_path:
        A string representing the absolute/relative path to the checkpoint file.
    """
    checkpoint_path = os.path.join(RESOURCES_DIR, checkpoint_path.split('resources')[-1])
    try:
        return torch.load(checkpoint_path, map_location='cpu')

    except FileNotFoundError:
        raise Exception('ERROR - Weights file missing: {}. '
                        'To download, please go to '
                        'https://20bn.com/licensing/sdk/evaluation and follow the '
                        'instructions.'.format(checkpoint_path))
