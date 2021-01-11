#!/usr/bin/env python
"""
This script helps prepare the necessary frames to be annotated for training a custom classifier for given data.

Usage:
  prepare_annotation.py --data_path=DATA_PATH
  prepare_annotation.py (-h | --help)

Options:
  --data_path=DATA_PATH     Complete or relative path to the data-set folder
"""

import os
import torch
import glob

from docopt import docopt
from os.path import join

from sense import engine
from sense import feature_extractors
from sense.finetuning import compute_frames_features

if __name__ == "__main__":
    # Parse argument
    args = docopt(__doc__)
    dataset_path = join(os.getcwd(), args['--data_path'])

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()

    # Remove internal padding for feature extraction and training
    checkpoint = torch.load('resources/backbone/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Create Inference Engine
    inference_engine = engine.InferenceEngine(feature_extractor, use_gpu=True)

    for split in ['train', 'valid']:
        print("\n" + "-" * 10 + f"Preparing videos in the {split}-set" + "-" * 10)
        for label in os.listdir(join(dataset_path, f'videos_{split}')):
            compute_frames_features(inference_engine, split, label, dataset_path)
    print('\nDone!')
