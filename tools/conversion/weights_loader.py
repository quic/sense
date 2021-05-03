import os
import torch

from sense.loading import load_backbone_model_from_config


def load_custom_classifier_weights(path_in):
    # Load backbone network according to config file
    backbone_model_config, backbone_weights = load_backbone_model_from_config(path_in)

    # Load custom classifier checkpoint
    weights_file = os.path.join(path_in, 'best_classifier.checkpoint')
    classifier_weights = torch.load(weights_file, map_location='cpu')

    all_weights = {
        'backbone': backbone_weights,
        'custom_classifier': classifier_weights,
    }

    return backbone_model_config, all_weights
