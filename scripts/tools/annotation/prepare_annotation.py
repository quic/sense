#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import torch
import glob
from realtimenet import feature_extractors
from realtimenet import engine
from realtimenet.finetuning import compute_features

# initialise network
# Load feature extractor
feature_extractor = feature_extractors.StridedInflatedEfficientNet()
# remove internal padding for feature extraction and training
checkpoint = torch.load('resources/backbone/strided_inflated_efficientnet.ckpt')
feature_extractor.load_state_dict(checkpoint)
feature_extractor.eval()

# Create inference engine
inference_engine = engine.InferenceEngine(feature_extractor, use_gpu=True)


folder = '/home/amercier/code/20bn-realtimenet/fitness_tl_benchmark/videos_train/Spider Man Pushup/'

out_folder = '/home/amercier/code/20bn-realtimenet/annotation/0/'
features_folder = out_folder + "features/"
frames_folder = out_folder + "frames/"
os.makedirs(out_folder, exist_ok=True)
os.makedirs(features_folder, exist_ok=True)
videos = glob.glob(folder + '*.mp4')
for e, video_path in enumerate(videos):
    print(f"processing video {e + 1}")
    path_frames = frames_folder + video_path.split("/")[-1].replace(".mp4", "")
    path_features = features_folder + video_path.split("/")[-1].replace(".mp4", ".npy")
    os.makedirs(path_frames, exist_ok=True)
    # Warning: if set a max batch size, you should not remove padding from model.
    compute_features(video_path, path_features, inference_engine,
                                 minimum_frames=0,  path_frames=path_frames, batch_size=64)
