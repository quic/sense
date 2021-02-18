#!/usr/bin/env python
"""
Web app for maintaining all of your video datasets:
- Setup new datasets with custom labels and temporal tags
- Record new videos (coming soon)
- Temporally annotate your videos with custom tags
- Train custom models using strong backbone networks (coming soon)
"""


import datetime
import glob
import json
import numpy as np
import os
import urllib

from flask import Flask
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for
from joblib import dump
from joblib import load
from sklearn.linear_model import LogisticRegression
from typing import List
from typing import Optional

from sense.engine import InferenceEngine
from sense.finetuning import compute_frames_features
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig


app = Flask(__name__)
app.secret_key = 'd66HR8dç"f_-àgjYYic*dh'

logreg: Optional[LogisticRegression] = None
inference_engine: Optional[InferenceEngine] = None
model_config: Optional[ModelConfig] = None

MODULE_DIR = os.path.dirname(__file__)
PROJECTS_OVERVIEW_CONFIG_FILE = os.path.join(MODULE_DIR, 'projects_config.json')

PROJECT_CONFIG_FILE = 'project_config.json'

SPLITS = ['train', 'valid']

SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', []),
    ModelConfig('StridedInflatedMobileNetV2', 'pro', []),
    ModelConfig('StridedInflatedEfficientNet', 'lite', []),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', []),
]


def _load_feature_extractor():
    global inference_engine
    global model_config

    if inference_engine is None:
        # Load weights
        selected_config, weights = get_relevant_weights(SUPPORTED_MODEL_CONFIGURATIONS)
        model_config = selected_config

        # Setup backbone network
        backbone_network = build_backbone_network(selected_config, weights['backbone'])

        # Create Inference Engine
        inference_engine = InferenceEngine(backbone_network, use_gpu=True)


def _extension_ok(filename):
    """ Returns `True` if the file has a valid image extension. """
    return '.' in filename and filename.rsplit('.', 1)[1] in ('png', 'jpg', 'jpeg', 'gif', 'bmp')


def _load_project_overview_config():
    if os.path.isfile(PROJECTS_OVERVIEW_CONFIG_FILE):
        with open(PROJECTS_OVERVIEW_CONFIG_FILE, 'r') as f:
            projects = json.load(f)
        return projects
    else:
        _write_project_overview_config({})
        return {}


def _write_project_overview_config(projects):
    with open(PROJECTS_OVERVIEW_CONFIG_FILE, 'w') as f:
        json.dump(projects, f, indent=2)


def _load_project_config(path):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def _write_project_config(path, config):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def _get_data_dir(dir_type: str, dataset_path: str, split: Optional[str] = None, subdirs: Optional[List[str]] = None):
    main_dir = f'{dir_type}_{split}' if split else dir_type
    subdirs = subdirs or []

    return os.path.join(dataset_path, main_dir, *subdirs)


def _get_videos_dir(dataset_path, split, label=None):
    subdirs = [label] if label else None
    return _get_data_dir('videos', dataset_path, split, subdirs)


def _get_frames_dir(dataset_path, split, label=None):
    subdirs = [label] if label else None
    return _get_data_dir('frames', dataset_path, split, subdirs)


def _get_features_dir(dataset_path, split, model: Optional[ModelConfig] = None, label=None):
    subdirs = None
    if model:
        subdirs = [model.combined_model_name]
        if label:
            subdirs.append(label)

    return _get_data_dir('features', dataset_path, split, subdirs)


def _get_tags_dir(dataset_path, split, label=None):
    subdirs = [label] if label else None
    return _get_data_dir('tags', dataset_path, split, subdirs)


def _get_logreg_dir(dataset_path, model: Optional[ModelConfig] = None, label=None):
    subdirs = None
    if model:
        subdirs = [model.combined_model_name]
        if label:
            subdirs.append(label)

    return _get_data_dir('logreg', dataset_path, subdirs=subdirs)


@app.route('/')
def projects_overview():
    """
    Home page of SenseStudio. Show the overview of all registered projects and check if their
    locations are still valid.
    """
    projects = _load_project_overview_config()

    # Check if project paths still exist
    for name, project in projects.items():
        project['exists'] = os.path.exists(project['path'])

    return render_template('projects_overview.html', projects=projects)


@app.route('/projects-list', methods=['POST'])
def projects_list():
    """
    Provide the current list of projects to external callers.
    """
    projects = _load_project_overview_config()
    return jsonify(projects)


@app.route('/project-config', methods=['POST'])
def project_config():
    """
    Provide the config for a given project.
    """
    data = request.json
    name = data['name']

    # Lookup project path
    projects = _load_project_overview_config()
    path = projects[name]['path']

    # Get config
    config = _load_project_config(path)
    return jsonify(config)


@app.route('/remove-project/<string:name>')
def remove_project(name):
    """
    Remove a given project from the config file and reload the overview page.
    """
    name = urllib.parse.unquote(name)
    projects = _load_project_overview_config()

    del projects[name]

    _write_project_overview_config(projects)

    return redirect(url_for('projects_overview'))


@app.route('/browse-directory', methods=['POST'])
def browse_directory():
    """
    Browse the local file system starting at the given path and provide the following information:
    - path_exists: If the given path exists
    - subdirs: The list of sub-directories at the given path
    """
    data = request.json
    path = data['path']

    subdirs = [d for d in glob.glob(f'{path}*') if os.path.isdir(d)] if os.path.isabs(path) else []

    return jsonify(path_exists=os.path.exists(path), subdirs=subdirs)


@app.route('/setup-project', methods=['POST'])
def setup_project():
    """
    Add a new project to the config file. Can also be used for updating an existing project.
    """
    data = request.form
    name = data['projectName']
    path = data['path']

    # Initialize project directory
    if not os.path.exists(path):
        os.mkdir(path)

    # Update project config
    try:
        # Check for existing config file
        config = _load_project_config(path)
        old_name = config['name']
        config['name'] = name
    except FileNotFoundError:
        # Setup new project config
        config = {
            'name': name,
            'date_created': datetime.date.today().isoformat(),
            'classes': {},
        }
        old_name = None

    _write_project_config(path, config)

    # Setup directory structure
    for split in SPLITS:
        videos_dir = _get_videos_dir(path, split)
        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

    # Update overall projects config file
    projects = _load_project_overview_config()

    if old_name and old_name in projects:
        del projects[old_name]

    projects[name] = {
        'path': path,
    }

    _write_project_overview_config(projects)

    return redirect(url_for('project_details', path=path))


@app.route('/project/<path:path>')
def project_details(path):
    """
    Show the details for the selected project.
    """
    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    config = _load_project_config(path)

    stats = {}
    for class_name, tags in config['classes'].items():
        stats[class_name] = {}
        for split in SPLITS:
            videos_dir = _get_videos_dir(path, split, class_name)
            tags_dir = _get_tags_dir(path, split, class_name)
            stats[class_name][split] = {
                'total': len(os.listdir(videos_dir)),
                'tagged': len(os.listdir(tags_dir)) if os.path.exists(tags_dir) else 0,
            }

    return render_template('project_details.html', config=config, path=path, stats=stats)


def _get_class_name_and_tags(form_data):
    """
    Extract 'className', 'tag1' and 'tag2' from the given form data and make sure that the tags
    are not empty or the same.
    """
    class_name = form_data['className']
    tag1 = form_data['tag1'] or f'{class_name}_tag1'
    tag2 = form_data['tag2'] or f'{class_name}_tag2'

    if tag2 == tag1:
        tag1 = f'{tag1}_1'
        tag2 = f'{tag2}_2'

    return class_name, tag1, tag2


@app.route('/add-class/<string:project>', methods=['POST'])
def add_class(project):
    """
    Add a new class to the given project.
    """
    project = urllib.parse.unquote(project)

    # Lookup project path
    projects = _load_project_overview_config()
    path = projects[project]['path']

    # Get class name and tags
    class_name, tag1, tag2 = _get_class_name_and_tags(request.form)

    # Update project config
    config = _load_project_config(path)
    config['classes'][class_name] = [tag1, tag2]
    _write_project_config(path, config)

    # Setup directory structure
    for split in SPLITS:
        videos_dir = _get_videos_dir(path, split, class_name)

        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

    return redirect(url_for("project_details", path=path))


@app.route('/edit-class/<string:project>/<string:class_name>', methods=['POST'])
def edit_class(project, class_name):
    """
    Edit the class name and tags for an existing class in the given project.
    """
    project = urllib.parse.unquote(project)
    class_name = urllib.parse.unquote(class_name)

    # Lookup project path
    projects = _load_project_overview_config()
    path = projects[project]['path']

    # Get new class name and tags
    new_class_name, new_tag1, new_tag2 = _get_class_name_and_tags(request.form)

    # Update project config
    config = _load_project_config(path)
    del config['classes'][class_name]
    config['classes'][new_class_name] = [new_tag1, new_tag2]
    _write_project_config(path, config)

    # Update directory names
    data_dirs = []
    for split in SPLITS:
        data_dirs.extend([
            _get_videos_dir(path, split),
            _get_frames_dir(path, split),
            _get_tags_dir(path, split),
        ])

        features_dir = _get_features_dir(path, split)
        data_dirs.extend([os.path.join(features_dir, model_dir) for model_dir in os.listdir(features_dir)])

    logreg_dir = _get_logreg_dir(path)
    data_dirs.extend([os.path.join(logreg_dir, model_dir) for model_dir in os.listdir(logreg_dir)])

    for base_dir in data_dirs:
        class_dir = os.path.join(base_dir, class_name)

        if os.path.exists(class_dir):
            new_class_dir = os.path.join(base_dir, new_class_name)
            os.rename(class_dir, new_class_dir)

    return redirect(url_for('project_details', path=path))


@app.route('/remove-class/<string:project>/<string:class_name>')
def remove_class(project, class_name):
    """
    Remove the given class from the config file of the given project. No data will be deleted.
    """
    project = urllib.parse.unquote(project)
    class_name = urllib.parse.unquote(class_name)

    # Lookup project path
    projects = _load_project_overview_config()
    path = projects[project]['path']

    # Update project config
    config = _load_project_config(path)
    del config['classes'][class_name]
    _write_project_config(path, config)

    return redirect(url_for("project_details", path=path))


@app.route('/annotate/<split>/<label>/<path:path>')
def show_video_list(split, label, path):
    """
    Show the list of videos for the given split, class label and project.
    If the necessary files for annotation haven't been prepared yet, this is done now.
    """
    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    split = urllib.parse.unquote(split)
    label = urllib.parse.unquote(label)

    # load feature extractor if needed
    _load_feature_extractor()

    videos_dir = _get_videos_dir(path, split, label)
    frames_dir = _get_frames_dir(path, split, label)
    features_dir = _get_features_dir(path, split, model_config, label)
    tags_dir = _get_tags_dir(path, split, label)
    logreg_dir = _get_logreg_dir(path, model_config, label)

    os.makedirs(logreg_dir, exist_ok=True)
    os.makedirs(tags_dir, exist_ok=True)

    # compute the features and frames missing
    compute_frames_features(inference_engine=inference_engine,
                            videos_dir=videos_dir,
                            frames_dir=frames_dir,
                            features_dir=features_dir)

    videos = os.listdir(frames_dir)
    videos.sort()

    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')
    if os.path.isfile(logreg_path):
        global logreg
        logreg = load(logreg_path)

    folder_id = zip(videos, list(range(len(videos))))
    return render_template('video_list.html', folders=folder_id, split=split, label=label, path=path)


@app.route('/prepare_annotation/<path:path>')
def prepare_annotation(path):
    """
    Prepare all files needed for annotating the videos in the given project.
    """
    dataset_path = f'/{urllib.parse.unquote(path)}'  # Make path absolute

    # load feature extractor if needed
    _load_feature_extractor()
    for split in SPLITS:
        print(f'\n\tPreparing videos in the {split}-set')

        for label in os.listdir(_get_videos_dir(dataset_path, split)):
            videos_dir = _get_videos_dir(dataset_path, split, label)
            frames_dir = _get_frames_dir(dataset_path, split, label)
            features_dir = _get_features_dir(dataset_path, split, model_config, label)

            compute_frames_features(inference_engine=inference_engine,
                                    videos_dir=videos_dir,
                                    frames_dir=frames_dir,
                                    features_dir=features_dir)

    return redirect(url_for("project_details", path=path))


@app.route('/annotate/<split>/<label>/<path:path>/<int:idx>')
def annotate(split, label, path, idx):
    """
    For the given class label, show all frames for annotating the selected video.
    """
    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    label = urllib.parse.unquote(label)
    split = urllib.parse.unquote(split)
    frames_dir = _get_frames_dir(path, split, label)
    features_dir = _get_features_dir(path, split, model_config, label)

    videos = os.listdir(frames_dir)
    videos.sort()

    features = np.load(os.path.join(features_dir, videos[idx] + ".npy"))
    features = features.mean(axis=(2, 3))

    if logreg is not None:
        classes = list(logreg.predict(features))
    else:
        classes = [-1] * len(features)

    # The list of images in the folder
    images = [image for image in glob.glob(os.path.join(frames_dir, videos[idx] + '/*'))
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


@app.route('/submit-annotation', methods=['POST'])
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

    frames_dir = _get_frames_dir(path, split, label)
    tags_dir = _get_tags_dir(path, split, label)
    description = {'file': f'{video}.mp4', 'fps': fps}

    out_annotation = os.path.join(tags_dir, f'{video}.json')
    time_annotation = []

    for frame_idx in range(int(data['n_images'])):
        time_annotation.append(int(data[f'{frame_idx}_tag']))

    description['time_annotation'] = time_annotation
    json.dump(description, open(out_annotation, 'w'))

    if next_frame_idx >= len(os.listdir(frames_dir)):
        return redirect(url_for('project_details', path=path))

    return redirect(url_for('annotate', split=split, label=label, path=path, idx=next_frame_idx))


@app.route('/train-logreg', methods=['POST'])
def train_logreg():
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    global logreg

    data = request.form  # a multi-dict containing POST data
    idx = int(data['idx'])
    path = data['path']
    split = data['split']
    label = data['label']

    features_dir = _get_features_dir(path, split, model_config, label)
    tags_dir = _get_tags_dir(path, split, label)
    logreg_dir = _get_logreg_dir(path, model_config, label)
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
        logreg = LogisticRegression(C=0.1, class_weight=class_weight)
        logreg.fit(X, y)
        dump(logreg, logreg_path)

    return redirect(url_for('annotate', split=split, label=label, path=path, idx=idx))


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/uploads/<path:img_path>')
def download_file(img_path):
    """
    Load an image from the given path.
    """
    img_path = f'/{urllib.parse.unquote(img_path)}'  # Make path absolute
    img_dir, img = os.path.split(img_path)
    return send_from_directory(img_dir, img, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
