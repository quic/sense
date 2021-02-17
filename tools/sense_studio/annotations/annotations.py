#!/usr/bin/env python

import glob
import json
import os
import urllib

import numpy as np
from flask import Blueprint
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for

from joblib import dump
from joblib import load
from os.path import join
from sklearn.linear_model import LogisticRegression

from sense.finetuning import compute_frames_features

from tools.sense_studio.utils import _extension_ok
from tools.sense_studio.utils import _load_feature_extractor
from tools.sense_studio.utils import _load_project_config

annotations_bp = Blueprint('annotations_bp', __name__, template_folder='templates/annotations')

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


@annotations_bp.route('/submit-annotation', methods=['POST'])
def submit_annotation():
    """
    Submit annotated tags for all frames and save them to a json file.
    """
    data = request.form  # a multi-dict containing POST data
    idx = int(data['idx'])
    fps = float(data['fps'])
    path = data['path']
    split = data['split']
    label = data['label']
    video = data['video']
    next_frame_idx = idx + 1

    tags_dir = join(path, f"tags_{split}", label)
    frames_dir = join(path, f"frames_{split}", label)
    description = {'file': video + ".mp4", 'fps': fps}

    out_annotation = os.path.join(tags_dir, video + ".json")
    time_annotation = []

    for frame_idx in range(int(data['n_images'])):
        time_annotation.append(int(data[f'{frame_idx}_tag']))

    description['time_annotation'] = time_annotation
    json.dump(description, open(out_annotation, 'w'))

    if next_frame_idx >= len(os.listdir(frames_dir)):
        return redirect(url_for('project_details', path=path))

    return redirect(url_for('.annotate', split=split, label=label, path=path, idx=next_frame_idx))


@annotations_bp.route('/train-logreg', methods=['POST'])
def train_logreg():
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """

    data = request.form  # a multi-dict containing POST data
    idx = int(data['idx'])
    path = data['path']
    split = data['split']
    label = data['label']

    tags_dir = join(path, f"tags_{split}", label)
    features_dir = join(path, f"features_{split}", label)
    logreg_dir = join(path, 'logreg', label)
    logreg_path = join(logreg_dir, 'logreg.joblib')

    annotations = os.listdir(tags_dir)
    class_weight = {0: 0.5}

    if annotations:
        features = [join(features_dir, x.replace('.json', '.npy')) for x in annotations]
        annotations = [join(tags_dir, x) for x in annotations]
        X = []
        y = []

        for feature in features:
            feature = np.load(feature)

            for f in feature:
                X.append(f.mean(axis=(1, 2)))

        for annotation in annotations:
            annotation = json.load(open(annotation, 'r'))['time_annotation']
            pos1 = np.where(np.array(annotation).astype(int) == 1)[0]

            if len(pos1) > 0:
                class_weight.update({1: 2})

                for p in pos1:
                    if p + 1 < len(annotation):
                        annotation[p + 1] = 1

            pos1 = np.where(np.array(annotation).astype(int) == 2)[0]

            if len(pos1) > 0:
                class_weight.update({2: 2})

                for p in pos1:
                    if p + 1 < len(annotation):
                        annotation[p + 1] = 2

            for a in annotation:
                y.append(a)

        X = np.array(X)
        y = np.array(y)
        logreg = LogisticRegression(C=0.1, class_weight=class_weight)
        logreg.fit(X, y)
        dump(logreg, logreg_path)

    return redirect(url_for('.annotate', split=split, label=label, path=path, idx=idx))


@annotations_bp.route('/uploads/<path:img_path>')
def download_file(img_path):
    """
    Load an image from the given path.
    """
    img_path = f'/{urllib.parse.unquote(img_path)}'  # Make path absolute
    img_dir, img = os.path.split(img_path)
    return send_from_directory(img_dir, img, as_attachment=True)