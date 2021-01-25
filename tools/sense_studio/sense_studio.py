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
from os.path import join
from sklearn.linear_model import LogisticRegression

from sense.finetuning import compute_frames_features


app = Flask(__name__)
app.secret_key = 'd66HR8dç"f_-àgjYYic*dh'

MODULE_DIR = os.path.dirname(__file__)
PROJECTS_OVERVIEW_CONFIG_FILE = os.path.join(MODULE_DIR, 'projects_config.json')

PROJECT_CONFIG_FILE = 'project_config.json'


def _load_feature_extractor():
    global inference_engine
    import torch
    from sense import engine
    from sense import feature_extractors
    if inference_engine is None:
        feature_extractor = feature_extractors.StridedInflatedEfficientNet()

        # Remove internal padding for feature extraction and training
        checkpoint = torch.load('resources/backbone/strided_inflated_efficientnet.ckpt')
        feature_extractor.load_state_dict(checkpoint)
        feature_extractor.eval()

        # Create Inference Engine
        inference_engine = engine.InferenceEngine(feature_extractor, use_gpu=True)


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


@app.route('/new-project-setup')
def new_project_setup():
    """
    Show the page for setting up a new project including name, path and class definitions.
    """
    return render_template('new_project_setup.html')


@app.route('/import-project/', defaults={'name': None, 'path': None})
@app.route('/import-project/<string:name>/<path:path>')
def import_project(name, path):
    """
    Show the page for importing an existing project. If name and path are given, this is used to
    update an existing project config, otherwise any existing directory can be selected from the
    local file system.
    """
    name = urllib.parse.unquote(name) if name else ''
    path = f'/{urllib.parse.unquote(path)}' if path else ''  # Make path absolute

    return render_template('import_project.html', name=name, path=path)


@app.route('/check-existing-project', methods=['POST'])
def check_existing_project():
    """
    Browse the local file system starting at the given path and provide the following information:
    - path_exsists: If the given path exists
    - classes: The list existing classes discovered in the videos_train directory
    - subdirs: The list of sub-directories at the given path
    """
    data = request.json
    path = data['path']

    subdirs = [d for d in glob.glob(f'{path}*') if os.path.isdir(d)]

    if not os.path.exists(path):
        return jsonify(path_exists=False, classes=[], subdirs=subdirs)

    train_dir = os.path.join(path, 'videos_train')
    classes = []
    if os.path.exists(train_dir):
        classes = os.listdir(train_dir)

    return jsonify(path_exists=True, classes=classes, subdirs=subdirs)


@app.route('/create-new-project', methods=['POST'])
def create_new_project():
    """
    Add a new project to the config file. Can also be used for updating an existing project.
    """
    data = request.form
    name = data['name']
    path = data['path']

    classes = {}
    class_idx = 0
    class_key = f'class{class_idx}'
    while class_key in data:
        class_name = data[class_key]
        if class_name:
            classes[class_name] = [
                data[f'{class_key}_tag{tag_idx}'] or f'{class_name}_{tag_idx}'
                for tag_idx in range(1, 3)
            ]

        class_idx += 1
        class_key = f'class{class_idx}'

    config = {
        'name': name,
        'date_created': datetime.date.today().isoformat(),
        'classes': classes,
    }

    # Initialize config file in project directory
    if not os.path.exists(path):
        os.mkdir(path)

    _write_project_config(path, config)

    # Setup directory structure
    for split in ['train', 'valid']:
        videos_dir = os.path.join(path, f'videos_{split}')
        if not os.path.exists(videos_dir):
            print(f'Creating {videos_dir}')
            os.mkdir(videos_dir)

        for class_name in classes:
            class_dir = os.path.join(videos_dir, class_name)

            if not os.path.exists(class_dir):
                print(f'Creating {class_dir}')
                os.mkdir(class_dir)

    # Update overall projects config file
    projects = _load_project_overview_config()
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
        for split in ['train', 'valid']:
            videos_path = os.path.join(path, f'videos_{split}', class_name)
            tags_path = os.path.join(path, f'tags_{split}', class_name)
            stats[class_name][split] = {
                'total': len(os.listdir(videos_path)),
                'tagged': len(os.listdir(tags_path)) if os.path.exists(tags_path) else 0,
            }

    return render_template('project_details.html', config=config, path=path, stats=stats)


@app.route('/add-class/<string:project>', methods=['POST'])
def add_class(project):
    """
    Add a new class to the given project.
    """
    # Lookup project path
    projects = _load_project_overview_config()
    path = projects[project]['path']

    # Get class name and tags
    data = request.form
    class_name = data['class']
    tag1 = data['tag1'] or f'{class_name}_tag1'
    tag2 = data['tag2'] or f'{class_name}_tag2'

    if tag2 == tag1:
        tag2 = f'{tag2}_2'

    # Update project config
    config = _load_project_config(path)
    config['classes'][class_name] = [tag1, tag2]
    _write_project_config(path, config)

    # Setup directory structure
    for split in ['train', 'valid']:
        videos_dir = os.path.join(path, f'videos_{split}')
        class_dir = os.path.join(videos_dir, class_name)
        os.mkdir(class_dir)

    return redirect(url_for("project_details", path=path))


@app.route('/annotate/<split>/<label>/<path:path>')
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
    _load_feature_extractor()
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


@app.route('/prepare_annotation/<path:path>')
def prepare_annotation(path):
    """
    Prepare all files needed for annotating the videos in the given project.
    """
    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute

    # load feature extractor if needed
    _load_feature_extractor()
    for split in ['train', 'valid']:
        print("\n" + "-" * 10 + f"Preparing videos in the {split}-set" + "-" * 10)
        for label in os.listdir(join(path, f'videos_{split}')):
            compute_frames_features(inference_engine, split, label, path)
    return redirect(url_for("project_details", path=path))


@app.route('/annotate/<split>/<label>/<path:path>/<int:idx>')
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
    logreg = None
    inference_engine = None

    app.run(debug=True)
