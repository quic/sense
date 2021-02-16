#!/usr/bin/env python

import glob
import os
import urllib
from os.path import join

import numpy as np
from flask import Blueprint
from flask import render_template
from joblib import load

from sense.finetuning import compute_frames_features
from tools.sense_studio.utils import _extension_ok
from tools.sense_studio.utils import _load_feature_extractor
from tools.sense_studio.utils import _load_project_config

annotations_bp = Blueprint('annotations_bp',  __name__, template_folder='templates/annotations')

logreg = None


@annotations_bp.route('/<split>/<label>/<path:path>/<int:idx>')
def annotate(split, label, path, idx):
    """
    For the given class label, show all frames for annotating the selected video.
    """
    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    label = urllib.parse.unquote(label)
    split = urllib.parse.unquote(split)
    frames_dir = join(path, f"frames_{split}", label)
    features_dir = join(path, f"features_{split}", label)

    videos = os.listdir(frames_dir)
    videos.sort()

    features = np.load(join(features_dir, videos[idx] + ".npy"))
    features = features.mean(axis=(2, 3))

    if logreg is not None:
        classes = list(logreg.predict(features))
    else:
        classes = [-1] * len(features)

    # The list of images in the folder
    images = [image for image in glob.glob(join(frames_dir, videos[idx] + '/*'))
              if _extension_ok(image)]

    # Add indexes
    images = sorted([(int(image.split('.')[0].split('/')[-1]), image) for image in images])  # TODO: Path ops?
    images = [[image, idx, _class] for (idx, image), _class in zip(images, classes)]

    # Read tags from config
    config = _load_project_config(path)
    tags = config['classes'][label]

    return render_template('frame_annotation.html', images=images, idx=idx, fps=16,
                           n_images=len(images), video_name=videos[idx],
                           split=split, label=label, path=path, tags=tags)


@annotations_bp.route('/<split>/<label>/<path:path>')
def show_video_list(split, label, path):
    """
    Show the list of videos for the given split, class label and project.
    If the necessary files for annotation haven't been prepared yet, this is done now.
    """

    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    split = urllib.parse.unquote((split))
    label = urllib.parse.unquote(label)
    frames_dir = join(path, f"frames_{split}", label)
    tags_dir = join(path, f"tags_{split}", label)
    logreg_dir = join(path, 'logreg', label)

    os.makedirs(logreg_dir, exist_ok=True)
    os.makedirs(tags_dir, exist_ok=True)

    # load feature extractor if needed
    inference_engine = _load_feature_extractor()
    # compute the features and frames missing.
    compute_frames_features(inference_engine, split, label, path)

    videos = os.listdir(frames_dir)
    videos.sort()

    logreg_path = join(logreg_dir, 'logreg.joblib')
    if os.path.isfile(logreg_path):
        global logreg
        logreg = load(logreg_path)

    folder_id = zip(videos, list(range(len(videos))))
    return render_template('video_list.html', folders=folder_id, split=split, label=label, path=path)

