import json
import os

from flask import g

MODULE_DIR = os.path.dirname(__file__)
PROJECTS_OVERVIEW_CONFIG_FILE = os.path.join(MODULE_DIR, 'projects_config.json')

PROJECT_CONFIG_FILE = 'project_config.json'

SPLITS = ['train', 'valid']


def _load_feature_extractor():

    import torch
    from sense import engine
    from sense import feature_extractors
    if g.inference_engine is None:
        feature_extractor = feature_extractors.StridedInflatedEfficientNet()

        # Remove internal padding for feature extraction and training
        checkpoint = torch.load('resources/backbone/strided_inflated_efficientnet.ckpt')
        feature_extractor.load_state_dict(checkpoint)
        feature_extractor.eval()

        # Create Inference Engine
        g.inference_engine = engine.InferenceEngine(feature_extractor, use_gpu=True)


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


def _lookup_project_path(project_name):
    projects = _load_project_overview_config()
    return projects[project_name]['path']


def _load_project_config(path):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def _write_project_config(path, config):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)



