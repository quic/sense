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
from sense.finetuning import compute_features


if __name__ == "__main__":
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
        print("\n" + "-"*10 + f"Preparing videos in the {split}-set" + "-"*10)
        for label in os.listdir(join(dataset_path, f'videos_{split}')):
            # Get data-set from path, given split and label
            folder = join(dataset_path, f'videos_{split}', label)

            # Create features and frames folders for the given split and label
            features_folder = join(dataset_path, f'features_{split}', label)
            frames_folder = join(dataset_path, f'frames_{split}', label)
            os.makedirs(features_folder, exist_ok=True)
            os.makedirs(frames_folder, exist_ok=True)

            # Loop through all videos for the given class-label
            videos = glob.glob(folder + '/*.mp4')
            for e, video_path in enumerate(videos):
                print(f"\r  Class: \"{label}\"  -->  Processing video {e + 1} / {len(videos)}", end="")
                path_frames = join(frames_folder, video_path.split("/")[-1].replace(".mp4", ""))
                path_features = join(features_folder, video_path.split("/")[-1].replace(".mp4", ".npy"))
                os.makedirs(path_frames, exist_ok=True)

                # WARNING: if set a max batch size, you should not remove padding from model.
                compute_features(video_path, path_features, inference_engine,
                                 minimum_frames=0,  path_frames=path_frames, batch_size=64)
            print()
    print('\nDone!')
