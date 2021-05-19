from collections import deque
from typing import List

import numpy as np


class PostProcessor:

    def __init__(self, indices=None):
        self.indices = indices

    def filter(self, predictions):
        if predictions is None:
            return predictions

        if self.indices:
            if len(self.indices) == 1:
                index = self.indices[0]
                return predictions[index]
            else:
                return [predictions[index] for index in self.indices]
        return predictions

    def postprocess(self, prediction):
        raise NotImplementedError

    def __call__(self, predictions):
        return self.postprocess(self.filter(predictions))


class PostprocessClassificationOutput(PostProcessor):

    def __init__(self, mapping_dict, smoothing=1, **kwargs):
        super().__init__(**kwargs)
        self.mapping = mapping_dict
        self.smoothing = smoothing
        assert smoothing >= 1
        self.buffer = deque(maxlen=smoothing)

    def postprocess(self, classif_output):
        if classif_output is not None:
            self.buffer.append(classif_output)

        if self.buffer:
            classif_output_smoothed = sum(self.buffer) / len(self.buffer)
        else:
            classif_output_smoothed = np.zeros(len(self.mapping))

        indices = classif_output_smoothed.argsort()

        return {
            'sorted_predictions': [(self.mapping[index], classif_output_smoothed[index])
                                   for index in indices[::-1]]
        }


class AggregatedPostProcessors(PostProcessor):
    """
    This class wraps a list of PostProcessors and stores their results under the same key in the output dictionary.
    """

    def __init__(self, post_processors: List[PostProcessor], out_key: str, **kwargs):
        """
        :param post_processors:
            List of PostProcessors whose results will be stored together in the output dictionary.
        :param out_key:
            Key for storing the aggregated results.
        """
        super().__init__(**kwargs)
        self.post_processors = post_processors
        self.out_key = out_key

    def postprocess(self, classif_output):
        output = {}
        for processor in self.post_processors:
            output.update(processor.postprocess(classif_output))

        return {self.out_key: output}


class TwoPositionsCounter(PostProcessor):
    """
    Count actions that are defined by alternating between two positions.
    """

    def __init__(self, pos0_idx: int, pos1_idx: int, threshold0: float, threshold1: float, out_key: str, **kwargs):
        super().__init__(**kwargs)
        self.pos0 = pos0_idx
        self.pos1 = pos1_idx
        self.threshold0 = threshold0
        self.threshold1 = threshold1
        self.count = 0
        self.current_position = 0
        self.out_key = out_key

    def postprocess(self, classif_output):
        if classif_output is not None:
            if self.current_position == 0:
                if classif_output[self.pos1] > self.threshold1:
                    self.current_position = 1
            else:
                if classif_output[self.pos0] > self.threshold0:
                    self.current_position = 0
                    self.count += 1

        return {self.out_key: self.count}


class EventCounter(PostProcessor):
    """
    Count how many times a certain event, tied to a specific model class, occurs.

    This class implements a locking mechanism that prevents counting multiple occurrences
    when the class probability remains above the provided threshold for multiple consecutive
    time-steps. More precisely, the class probability should first decrease below half the
    provided threshold before another occurrence can be counted. In other words, this object
    detects and counts probability spikes.
    """

    def __init__(self, key, key_idx, threshold, out_key=None, **kwargs):
        """
        :param key:
            The name of the class that should be counted.
        :param key_idx:
            The index of the counted class in the predicted probability tensor.
        :param threshold:
            The threshold that should be reached for a probability spike to be counted.
        :param out_key:
            Optional key for storing the output. If none is given, the input key will be used instead.
        """
        super().__init__(**kwargs)
        self.key = key
        self.key_idx = key_idx
        self.threshold = threshold
        self.out_key = out_key or key
        self.count = 0
        self.active = False

    def postprocess(self, classif_output):
        if classif_output is not None:
            if self.active and classif_output[self.key_idx] < (self.threshold / 2.):
                self.active = False
            elif not self.active and classif_output[self.key_idx] > self.threshold:
                self.active = True
                self.count += 1
        return {self.out_key: self.count}
