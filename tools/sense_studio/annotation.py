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
from sklearn.linear_model import LogisticRegression

from sense.finetuning import compute_frames_features
from tools.sense_studio import utils

annotation_bp = Blueprint('annotation_bp', __name__)


@annotation_bp.route('/<split>/<label>/<path:path>')
def show_video_list(split, label, path):
    """
    Show the list of videos for the given split, class label and project.
    If the necessary files for annotation haven't been prepared yet, this is done now.
    """
    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    split = urllib.parse.unquote(split)
    label = urllib.parse.unquote(label)

    # load feature extractor
    inference_engine, model_config = utils.load_feature_extractor()

    videos_dir = utils.get_videos_dir(path, split, label)
    frames_dir = utils.get_frames_dir(path, split, label)
    features_dir = utils.get_features_dir(path, split, model_config, label)
    tags_dir = utils.get_tags_dir(path, split, label)
    logreg_dir = utils.get_logreg_dir(path, model_config, label)

    os.makedirs(logreg_dir, exist_ok=True)
    os.makedirs(tags_dir, exist_ok=True)

    # compute the features and frames missing
    compute_frames_features(inference_engine=inference_engine,
                            videos_dir=videos_dir,
                            frames_dir=frames_dir,
                            features_dir=features_dir)

    videos = os.listdir(frames_dir)
    videos.sort()

    tagged_list = set(os.listdir(tags_dir))
    tagged = [f'{video}.json' in tagged_list for video in videos]

    video_list = zip(videos, tagged, list(range(len(videos))))
    return render_template('video_list.html', video_list=video_list, split=split, label=label, path=path)


@annotation_bp.route('/prepare-annotation/<path:path>')
def prepare_annotation(path):
    """
    Prepare all files needed for annotating the videos in the given project.
    """
    dataset_path = f'/{urllib.parse.unquote(path)}'  # Make path absolute

    # load feature extractor
    inference_engine, model_config = utils.load_feature_extractor()

    for split in utils.SPLITS:
        print(f'\n\tPreparing videos in the {split}-set')

        for label in os.listdir(utils.get_videos_dir(dataset_path, split)):
            videos_dir = utils.get_videos_dir(dataset_path, split, label)
            frames_dir = utils.get_frames_dir(dataset_path, split, label)
            features_dir = utils.get_features_dir(dataset_path, split, model_config, label)

            compute_frames_features(inference_engine=inference_engine,
                                    videos_dir=videos_dir,
                                    frames_dir=frames_dir,
                                    features_dir=features_dir)
    return redirect(url_for("project_details", path=path))


@annotation_bp.route('/<split>/<label>/<path:path>/<int:idx>')
def annotate(split, label, path, idx):
    """
    For the given class label, show all frames for annotating the selected video.
    """
    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    label = urllib.parse.unquote(label)
    split = urllib.parse.unquote(split)

    _, model_config = utils.load_feature_extractor()

    frames_dir = utils.get_frames_dir(path, split, label)
    features_dir = utils.get_features_dir(path, split, model_config, label)
    logreg_dir = utils.get_logreg_dir(path, model_config, label)

    videos = os.listdir(frames_dir)
    videos.sort()

    features = np.load(os.path.join(features_dir, videos[idx] + ".npy"))
    features = features.mean(axis=(2, 3))

    # Load logistic regression model if available
    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')
    if os.path.isfile(logreg_path):
        logreg = load(logreg_path)
        classes = list(logreg.predict(features))
    else:
        classes = [-1] * len(features)

    # The list of images in the folder
    images = [image for image in glob.glob(os.path.join(frames_dir, videos[idx] + '/*'))
              if utils.is_image_file(image)]

    # Add indexes
    images = sorted([(int(image.split('.')[0].split('/')[-1]), image) for image in images])  # TODO: Path ops?
    images = [[image, idx, _class] for (idx, image), _class in zip(images, classes)]

    # Read tags from config
    config = utils.load_project_config(path)
    tags = config['classes'][label]

    return render_template('frame_annotation.html', images=images, idx=idx, fps=16,
                           n_images=len(images), video_name=videos[idx],
                           split=split, label=label, path=path, tags=tags)


@annotation_bp.route('/submit-annotation', methods=['POST'])
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

    frames_dir = utils.get_frames_dir(path, split, label)
    tags_dir = utils.get_tags_dir(path, split, label)
    description = {'file': f'{video}.mp4', 'fps': fps}

    out_annotation = os.path.join(tags_dir, f'{video}.json')
    time_annotation = []

    for frame_idx in range(int(data['n_images'])):
        time_annotation.append(int(data[f'{frame_idx}_tag']))

    description['time_annotation'] = time_annotation
    json.dump(description, open(out_annotation, 'w'))

    if next_frame_idx >= len(os.listdir(frames_dir)):
        return redirect(url_for('project_details', path=path))

    return redirect(url_for('.annotate', split=split, label=label, path=path, idx=next_frame_idx))


@annotation_bp.route('/train-logreg', methods=['POST'])
def train_logreg():
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    data = request.form  # a multi-dict containing POST data
    idx = int(data['idx'])
    path = data['path']
    split = data['split']
    label = data['label']

    _, model_config = utils.load_feature_extractor()

    features_dir = utils.get_features_dir(path, split, model_config, label)
    tags_dir = utils.get_tags_dir(path, split, label)
    logreg_dir = utils.get_logreg_dir(path, model_config, label)
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

        if len(class_weight) > 1:
            logreg = LogisticRegression(C=0.1, class_weight=class_weight)
            logreg.fit(X, y)
            dump(logreg, logreg_path)

    return redirect(url_for('.annotate', split=split, label=label, path=path, idx=idx))


@annotation_bp.route('/uploads/<path:img_path>')
def download_file(img_path):
    """
    Load an image from the given path.
    """
    img_path = f'/{urllib.parse.unquote(img_path)}'  # Make path absolute
    img_dir, img = os.path.split(img_path)
    return send_from_directory(img_dir, img, as_attachment=True)
