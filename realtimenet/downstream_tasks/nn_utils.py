import torch.nn as nn


class Pipe(nn.Module):

    def __init__(self, feature_extractor, feature_converter):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_converter = feature_converter

    def forward(self, video_tensor):
        feature = self.feature_extractor(video_tensor)
        if isinstance(self.feature_converter, list):
            return [convert(feature) for convert in self.feature_converter]
        return self.feature_converter(feature)

    @property
    def expected_frame_size(self):
        return self.feature_extractor.expected_frame_size

    @property
    def fps(self):
        return self.feature_extractor.fps

    @property
    def step_size(self):
        return self.feature_extractor.step_size

    def preprocess(self, video):
        return self.feature_extractor.preprocess(video)


class LogisticRegression(nn.Sequential):

    def __init__(self, num_in, num_out):
        super(LogisticRegression, self).__init__(
            nn.Linear(num_in, num_out),
            nn.Softmax(dim=-1)
        )

