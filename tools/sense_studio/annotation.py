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
from joblib import load
from natsort import natsorted
from natsort import ns

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
                            project_path=path,
                            videos_dir=videos_dir,
                            frames_dir=frames_dir,
                            features_dir=features_dir)

    videos = os.listdir(frames_dir)
    videos = natsorted(videos, alg=ns.IC)

    tagged_list = set(os.listdir(tags_dir))
    tagged = [f'{video}.json' in tagged_list for video in videos]

    video_list = zip(videos, tagged, list(range(len(videos))))
    return render_template('video_list.html', video_list=video_list, split=split, label=label, path=path,
                           project=project)


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
                                    project_path=dataset_path,
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

    # The list of images in the folder
    images = [image for image in glob.glob(os.path.join(frames_dir, videos[idx], '*'))
              if utils.is_image_file(image)]
    classes = [-1] * len(images)

    # Load logistic regression model if available and assisted tagging is enabled
    if utils.get_project_setting(path, 'assisted_tagging'):
        logreg_path = os.path.join(logreg_dir, 'logreg.joblib')
        features_path = os.path.join(features_dir, f'{videos[idx]}.npy')
        if os.path.isfile(logreg_path) and os.path.isfile(features_path):
            logreg = load(logreg_path)
            features = np.load(features_path).mean(axis=(2, 3))
            classes = list(logreg.predict(features))

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

    return render_template('frame_annotation.html', images=images, annotations=annotations, idx=idx, fps=16,
                           n_images=len(images), video_name=videos[idx], project_config=config,
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
    if utils.get_project_setting(path, 'assisted_tagging'):
        inference_engine, model_config = utils.load_feature_extractor(path)
        videos_dir = directories.get_videos_dir(path, split, label)
        frames_dir = directories.get_frames_dir(path, split, label)
        features_dir = directories.get_features_dir(path, split, model_config, label=label)

        # Compute the respective frames and features
        compute_frames_features(inference_engine=inference_engine,
                                project_path=path,
                                videos_dir=videos_dir,
                                frames_dir=frames_dir,
                                features_dir=features_dir)

        # Re-train the logistic regression model
        utils.train_logreg(path=path, split=split, label=label)

    if next_frame_idx >= len(os.listdir(frames_dir)):
        return redirect(url_for('.show_video_list', project=project, split=split, label=label))

    return redirect(url_for('.annotate', split=split, label=label, project=project, idx=next_frame_idx))


@annotation_bp.route('/uploads/<string:project>/<string:split>/<string:label>/<string:video_name>/<string:img_file>')
def download_file(project, split, label, video_name, img_file):
    """
    Load an image from the given path.
    """
    dataset_path = utils.lookup_project_path(project)
    img_dir = os.path.join(directories.get_frames_dir(dataset_path, split, label), video_name)
    return send_from_directory(img_dir, img_file, as_attachment=True)
