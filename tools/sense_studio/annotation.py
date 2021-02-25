import glob
import json
import numpy as np
import os
import urllib

from flask import Blueprint
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for

from joblib import dump
from joblib import load
from natsort import natsorted
from os.path import join
from sklearn.linear_model import LogisticRegression

from sense.finetuning import compute_frames_features
from tools.sense_studio import utils


annotation_bp = Blueprint('annotation_bp', __name__)


@annotation_bp.route('/<string:project>/<string:split>/<string:label>')
def show_video_list(project, split, label):
    """
    Show the list of videos for the given split, class label and project.
    If the necessary files for annotation haven't been prepared yet, this is done now.
    """
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)
    split = urllib.parse.unquote(split)
    label = urllib.parse.unquote(label)
    frames_dir = join(path, f"frames_{split}", label)
    tags_dir = join(path, f"tags_{split}", label)
    logreg_dir = join(path, 'logreg', label)
    labels = utils.get_class_labels(path)

    os.makedirs(logreg_dir, exist_ok=True)
    os.makedirs(tags_dir, exist_ok=True)

    # load feature extractor
    inference_engine = utils.load_feature_extractor()
    # compute the features and frames missing.
    compute_frames_features(inference_engine, split, label, path)

    videos = os.listdir(frames_dir)
    videos = natsorted(videos)

    tagged_list = set(os.listdir(tags_dir))
    tagged = [f'{video}.json' in tagged_list for video in videos]

    video_list = zip(videos, tagged, list(range(len(videos))))
    return render_template('video_list.html', video_list=video_list, split=split, label=label, path=path,
                           project=project, labels=labels)


@annotation_bp.route('/prepare-annotation/<string:project>')
def prepare_annotation(project):
    """
    Prepare all files needed for annotating the videos in the given project.
    """
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)

    # load feature extractor
    inference_engine = utils.load_feature_extractor()
    for split in utils.SPLITS:
        print("\n" + "-" * 10 + f"Preparing videos in the {split}-set" + "-" * 10)
        for label in os.listdir(join(path, f'videos_{split}')):
            compute_frames_features(inference_engine, split, label, path)

    return redirect(url_for("project_details", project=project))


@annotation_bp.route('/<string:project>/<string:split>/<string:label>/<int:idx>')
def annotate(project, split, label, idx):
    """
    For the given class label, show all frames for annotating the selected video.
    """
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)
    label = urllib.parse.unquote(label)
    split = urllib.parse.unquote(split)
    frames_dir = join(path, f"frames_{split}", label)
    features_dir = join(path, f"features_{split}", label)
    tags_dir = join(path, f"tags_{split}", label)
    logreg_dir = join(path, 'logreg', label)

    videos = os.listdir(frames_dir)
    videos.sort()

    features = np.load(join(features_dir, videos[idx] + ".npy"))
    features = features.mean(axis=(2, 3))

    # Load logistic regression model if available
    logreg_path = join(logreg_dir, 'logreg.joblib')
    if os.path.isfile(logreg_path):
        logreg = load(logreg_path)
        classes = list(logreg.predict(features))
    else:
        classes = [-1] * len(features)

    # The list of images in the folder
    images = [image for image in glob.glob(join(frames_dir, videos[idx] + '/*'))
              if utils.is_image_file(image)]

    # Natural sort images, so that they are sorted by number
    images = natsorted(images)
    # Extract image file name (without full path) and include class label
    images = [(os.path.basename(image), _class) for image, _class in zip(images, classes)]

    # Load existing annotations
    annotations = []
    annotations_file = join(tags_dir, f'{videos[idx]}.json')
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            annotations = data['time_annotation']

    # Read tags from config
    config = utils.load_project_config(path)
    tags = config['classes'][label]

    return render_template('frame_annotation.html', images=images, annotations=annotations, idx=idx, fps=16,
                           n_images=len(images), video_name=videos[idx],
                           split=split, label=label, path=path, tags=tags, project=project, n_videos=len(videos))


@annotation_bp.route('/submit-annotation', methods=['POST'])
def submit_annotation():
    """
    Submit annotated tags for all frames and save them to a json file.
    """
    data = request.form  # a multi-dict containing POST data
    idx = int(data['idx'])
    fps = float(data['fps'])
    path = data['path']
    project = data['project']
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

    with open(out_annotation, 'w') as f:
        json.dump(description, f)

    if next_frame_idx >= len(os.listdir(frames_dir)):
        return redirect(url_for('project_details', project=project))

    return redirect(url_for('.annotate', split=split, label=label, project=project, idx=next_frame_idx))


@annotation_bp.route('/train-logreg', methods=['POST'])
def train_logreg():
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    data = request.form  # a multi-dict containing POST data
    idx = int(data['idx'])
    path = data['path']
    project = data['project']
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
            with open(annotation, 'r') as f:
                annotation = json.load(f)['time_annotation']

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

        if len(class_weight) > 1:
            logreg = LogisticRegression(C=0.1, class_weight=class_weight)
            logreg.fit(X, y)
            dump(logreg, logreg_path)

    return redirect(url_for('.annotate', split=split, label=label, project=project, idx=idx))


@annotation_bp.route('/uploads/<string:project>/<string:split>/<string:label>/<string:video_name>/<string:img_file>')
def download_file(project, split, label, video_name, img_file):
    """
    Load an image from the given path.
    """
    img_dir = utils.lookup_project_path(project) + f'/frames_{split}/{label}/{video_name}'
    return send_from_directory(img_dir, img_file, as_attachment=True)
