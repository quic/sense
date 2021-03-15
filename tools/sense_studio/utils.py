import json
import numpy as np
import os

from joblib import dump
from sense.engine import InferenceEngine
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig
from sklearn.linear_model import LogisticRegression

MODULE_DIR = os.path.dirname(__file__)
PROJECTS_OVERVIEW_CONFIG_FILE = os.path.join(MODULE_DIR, 'projects_config.json')

PROJECT_CONFIG_FILE = 'project_config.json'

SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', []),
    ModelConfig('StridedInflatedMobileNetV2', 'pro', []),
    ModelConfig('StridedInflatedEfficientNet', 'lite', []),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', []),
]


def load_feature_extractor(project_path):
    # Load weights
    model_config, weights = get_relevant_weights(SUPPORTED_MODEL_CONFIGURATIONS)

    # Setup backbone network
    backbone_network = build_backbone_network(model_config, weights['backbone'])

    # Create Inference Engine
    use_gpu = get_project_setting(project_path, 'use_gpu')
    inference_engine = InferenceEngine(backbone_network, use_gpu=use_gpu)

    return inference_engine, model_config


def is_image_file(filename):
    """ Returns `True` if the file has a valid image extension. """
    return '.' in filename and filename.rsplit('.', 1)[1] in ('png', 'jpg', 'jpeg', 'gif', 'bmp')


def load_project_overview_config():
    if os.path.isfile(PROJECTS_OVERVIEW_CONFIG_FILE):
        with open(PROJECTS_OVERVIEW_CONFIG_FILE, 'r') as f:
            projects = json.load(f)
        return projects
    else:
        write_project_overview_config({})
        return {}


def write_project_overview_config(projects):
    with open(PROJECTS_OVERVIEW_CONFIG_FILE, 'w') as f:
        json.dump(projects, f, indent=2)


def lookup_project_path(project_name):
    projects = load_project_overview_config()
    return projects[project_name]['path']


def load_project_config(path):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def write_project_config(path, config):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_class_name_and_tags(form_data):
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


def get_project_setting(path, setting):
    config = load_project_config(path)
    return config.get(setting, False)


def toggle_project_setting(path, setting):
    config = load_project_config(path)
    current_status = config.get(setting, False)

    new_status = not current_status
    config[setting] = new_status
    write_project_config(path, config)

    return new_status


def train_logreg(path, split, label):
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    tags_dir = os.path.join(path, f"tags_{split}", label)
    features_dir = os.path.join(path, f"features_{split}", label)
    logreg_dir = os.path.join(path, 'logreg', label)
    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')

    annotations = os.listdir(tags_dir)
    class_weight = {0: 0.5}

    if annotations:
        features = [os.path.join(features_dir, x.replace('.json', '.npy')) for x in annotations]
        annotations = [os.path.join(tags_dir, x) for x in annotations]
        x = []
        y = []

        for feature in features:
            feature = np.load(feature)

            for f in feature:
                x.append(f.mean(axis=(1, 2)))

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

        x = np.array(x)
        y = np.array(y)

        if len(class_weight) > 1:
            logreg = LogisticRegression(C=0.1, class_weight=class_weight)
            logreg.fit(x, y)
            dump(logreg, logreg_path)
