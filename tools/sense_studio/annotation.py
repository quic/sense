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
from natsort import ns
from sklearn.linear_model import LogisticRegression

from sense import SPLITS
from sense.finetuning import compute_frames_features
from tools import directories
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

    # load feature extractor
    inference_engine, model_config = utils.load_feature_extractor(path)

    videos_dir = directories.get_videos_dir(path, split, label)
    frames_dir = directories.get_frames_dir(path, split, label)
    features_dir = directories.get_features_dir(path, split, model_config, label=label)
    tags_dir = directories.get_tags_dir(path, split, label)
    logreg_dir = directories.get_logreg_dir(path, model_config, label)

    os.makedirs(logreg_dir, exist_ok=True)
    os.makedirs(tags_dir, exist_ok=True)

    # compute the features and frames missing
    compute_frames_features(inference_engine=inference_engine,
                            videos_dir=videos_dir,
                            frames_dir=frames_dir,
                            features_dir=features_dir)

    videos = os.listdir(frames_dir)
    videos = natsorted(videos, alg=ns.IC)

    tagged_list = set(os.listdir(tags_dir))
    tagged = [f'{video}.json' in tagged_list for video in videos]

    num_videos = len(videos)
    num_tagged = len(tagged_list)
    num_untagged = num_videos - num_tagged

    video_list = zip(videos, tagged, list(range(len(videos))))
    return render_template('video_list.html', video_list=video_list, split=split, label=label, path=path,
                           project=project, num_videos=num_videos, num_tagged=num_tagged, num_untagged=num_untagged)


@annotation_bp.route('/prepare-annotation/<string:project>')
def prepare_annotation(project):
    """
    Prepare all files needed for annotating the videos in the given project.
    """
    project = urllib.parse.unquote(project)
    dataset_path = utils.lookup_project_path(project)

    # load feature extractor
    inference_engine, model_config = utils.load_feature_extractor(dataset_path)
    for split in SPLITS:
        print(f'\n\tPreparing videos in the {split}-set')

        for label in os.listdir(directories.get_videos_dir(dataset_path, split)):
            videos_dir = directories.get_videos_dir(dataset_path, split, label)
            frames_dir = directories.get_frames_dir(dataset_path, split, label)
            features_dir = directories.get_features_dir(dataset_path, split, model_config, label=label)

            compute_frames_features(inference_engine=inference_engine,
                                    videos_dir=videos_dir,
                                    frames_dir=frames_dir,
                                    features_dir=features_dir)

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

    _, model_config = utils.load_feature_extractor(path)

    frames_dir = directories.get_frames_dir(path, split, label)
    features_dir = directories.get_features_dir(path, split, model_config, label=label)
    tags_dir = directories.get_tags_dir(path, split, label)
    logreg_dir = directories.get_logreg_dir(path, model_config, label)

    videos = os.listdir(frames_dir)
    videos.sort()

    features = np.load(os.path.join(features_dir, f'{videos[idx]}.npy'))
    features = features.mean(axis=(2, 3))

    # Load logistic regression model if available
    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')
    if os.path.isfile(logreg_path):
        logreg = load(logreg_path)
        classes = list(logreg.predict(features))
    else:
        classes = [-1] * len(features)

    # The list of images in the folder
    images = [image for image in glob.glob(os.path.join(frames_dir, videos[idx], '*'))
              if utils.is_image_file(image)]

    # Natural sort images, so that they are sorted by number
    images = natsorted(images, alg=ns.IC)
    # Extract image file name (without full path) and include class label
    images = [(os.path.basename(image), _class) for image, _class in zip(images, classes)]

    # Load existing annotations
    annotations = []
    annotations_file = os.path.join(tags_dir, f'{videos[idx]}.json')
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            annotations = data['time_annotation']

    # Read tags from config
    config = utils.load_project_config(path)
    tags = config['classes'][label]
    show_logreg = config.get('show_logreg', False)

    return render_template('frame_annotation.html', images=images, annotations=annotations, idx=idx, fps=16,
                           n_images=len(images), video_name=videos[idx], show_logreg=show_logreg,
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

    frames_dir = directories.get_frames_dir(path, split, label)
    tags_dir = directories.get_tags_dir(path, split, label)
    description = {'file': f'{video}.mp4', 'fps': fps}

    out_annotation = os.path.join(tags_dir, f'{video}.json')
    time_annotation = []

    for frame_idx in range(int(data['n_images'])):
        time_annotation.append(int(data[f'{frame_idx}_tag']))

    description['time_annotation'] = time_annotation

    with open(out_annotation, 'w') as f:
        json.dump(description, f, indent=2)

    # Automatic re-training of the logistic regression model
    if utils.get_project_setting(path, 'show_logreg'):
        train_logreg(path, split, label)

    if next_frame_idx >= len(os.listdir(frames_dir)):
        return redirect(url_for('.show_video_list', project=project, split=split, label=label))

    return redirect(url_for('.annotate', split=split, label=label, project=project, idx=next_frame_idx))


def train_logreg(path, split, label):
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    _, model_config = utils.load_feature_extractor(path)

    features_dir = directories.get_features_dir(path, split, model_config, label=label)
    tags_dir = directories.get_tags_dir(path, split, label)
    logreg_dir = directories.get_logreg_dir(path, model_config, label)
    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')

    annotations = os.listdir(tags_dir)
    class_weight = {0: 0.5}

    if annotations:
        features = [os.path.join(features_dir, x.replace('.json', '.npy')) for x in annotations]
        annotations = [os.path.join(tags_dir, x) for x in annotations]
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


@annotation_bp.route('/uploads/<string:project>/<string:split>/<string:label>/<string:video_name>/<string:img_file>')
def download_file(project, split, label, video_name, img_file):
    """
    Load an image from the given path.
    """
    dataset_path = utils.lookup_project_path(project)
    img_dir = os.path.join(directories.get_frames_dir(dataset_path, split, label), video_name)
    return send_from_directory(img_dir, img_file, as_attachment=True)
