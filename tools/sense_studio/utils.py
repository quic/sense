import json
import os
import torch

from sense import backbone_networks
from sense import engine

MODULE_DIR = os.path.dirname(__file__)
PROJECTS_OVERVIEW_CONFIG_FILE = os.path.join(MODULE_DIR, 'projects_config.json')

PROJECT_CONFIG_FILE = 'project_config.json'

SPLITS = ['train', 'valid']


def load_feature_extractor(project_path):
    feature_extractor = backbone_networks.StridedInflatedEfficientNet()

    # Remove internal padding for feature extraction and training
    checkpoint = torch.load('resources/backbone/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Create Inference Engine
    use_gpu = get_gpu_status(project_path)
    inference_engine = engine.InferenceEngine(feature_extractor, use_gpu=use_gpu)

    return inference_engine


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


def get_class_labels(path):
    """
    Extract class names from the config.
    """
    config = load_project_config(path)
    return config['classes'].keys()


def get_gpu_status(path):
    config = load_project_config(path)
    return config.get('use_gpu', False)


def toggle_gpu_status(path):
    config = load_project_config(path)
    current_status = config.get('use_gpu', False)
    new_status = not current_status

    config['use_gpu'] = new_status
    write_project_config(path, config)

    return new_status
