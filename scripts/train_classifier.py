#!/usr/bin/env python
"""
Real time detection of 30 hand gestures.

Usage:
  train_classifier.py    --path_in=PATH
                         [--use_gpu]
  train_classifier.py (-h | --help)

Options:
  --path_in=PATH              path to the dataset folder following the structure described in the readme
"""
import torch
from docopt import docopt
import os
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from realtimenet.downstream_tasks.nn_utils import LogisticRegression as TorchLogisticRegression

import realtimenet.display
from realtimenet import camera
from realtimenet import engine
from realtimenet import feature_extractors
import json


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    path_in = args['--path_in']
    use_gpu = args['--use_gpu']



    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    checkpoint = torch.load('resources/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Concatenate feature extractor and met converter
    net = feature_extractor

    # Create inference engine, video streaming and display instances
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)


    postprocessor = []
    display_ops = []


    # list the videos files
    videos_dir = os.path.join(path_in, "videos")
    classes = os.listdir(videos_dir)
    classes = [x for x in classes if not x.startswith('.')]
    print(classes)
    number_videos_found = len(glob.glob(os.path.join(videos_dir, "*", "*.mp4")))
    number_videos_processed = 0
    print(f"Found {number_videos_found} videos to process")

    for label in classes:
        videos = os.listdir(os.path.join(videos_dir, label))
        videos = [x for x in videos if not x.startswith('.')]
        for video in videos:
            print(f"extract features from video {number_videos_processed} / {number_videos_found}")
            # reset the buffer
            net.train()
            net.eval()
            number_videos_processed += 1
            file_path = os.path.join(videos_dir, label, video)
            path_out = os.path.join(path_in, "features", label, video.replace(".mp4", ".npy"))

            video_source = camera.VideoSource(camera_id=None,
                                              size=inference_engine.expected_frame_size,
                                              filename=file_path)
            frames = []
            features = []
            while True:
                images = video_source.get_image()
                if images is None:
                    break
                else:
                    image, image_rescaled = images
                    frames.append(image_rescaled)
                    if len(frames) == net.step_size:
                        clip = np.array([frames]).astype(np.float32)
                        frames = []
                        predictions = inference_engine.process_clip(clip)
                        features.append(predictions)
            if len(features) < 13:
                print("video too short")
            else:
                features = np.array(features[12:])
                os.makedirs(os.path.dirname((path_out)), exist_ok=True)
                np.save(path_out, features)

    features_dir =  os.path.join(path_in, "features")

    y = []
    X = []
    class2int = {x:e for e,x in enumerate(classes)}
    for label in classes:
        features = os.listdir(os.path.join(features_dir, label))
        # used to remove .DSstore files on mac
        features = [x for x in features if not x.startswith('.')]
        for feature in features:
            feature = np.load(os.path.join(features_dir, label, feature))
            X = X + list(feature)
            y += [class2int[label]] * feature.shape[0]
    X = np.array(X)
    y = np.array(y)

    clf = LogisticRegression(random_state=0, multi_class="multinomial")
    clf.fit(X,y)
    weights = clf.coef_
    bias = clf.intercept_

    net = TorchLogisticRegression(1280, len(classes))
    new_state_dict = net.state_dict()
    new_state_dict['0.weight'] = torch.Tensor(weights)
    new_state_dict['0.bias'] = torch.Tensor(bias)
    torch.save(new_state_dict, os.path.join(path_in, "classifier.checkpoint"))
    json.dumps(class2int, open("class2int.json", "w"))


