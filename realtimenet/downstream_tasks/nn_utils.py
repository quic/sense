import torch.nn as nn


class Pipe(nn.Module):

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

    def __init__(self, num_in, num_out, global_average_pooling=True):
        super(LogisticRegression, self).__init__(
            nn.Linear(num_in, num_out),
            # nn.Softmax(dim=-1)
        )
        self.global_average_pooling = global_average_pooling

    def forward(self, input_tensor):
        if self.global_average_pooling:
            input_tensor = input_tensor.mean(dim=-1).mean(dim=-1)
        return super().forward(input_tensor)
