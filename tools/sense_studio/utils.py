import json
import numpy as np
import os

from joblib import dump
from sklearn.linear_model import LogisticRegression

from sense.engine import InferenceEngine
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig
from tools import directories
from tools.sense_studio.project_utils import get_project_setting

SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', []),
    ModelConfig('StridedInflatedMobileNetV2', 'pro', []),
    ModelConfig('StridedInflatedEfficientNet', 'lite', []),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', []),
]


def get_available_backbone_models():
    """
    Get list of combined model names for all backbone models for which weights can be found in the local resources
    folder.
    """
    return [model.combined_model_name for model in SUPPORTED_MODEL_CONFIGURATIONS if model.weights_available()]


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


def train_logreg(path, split, label):
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    _, model_config = load_feature_extractor(path)

    features_dir = directories.get_features_dir(path, split, model_config, label=label)
    tags_dir = directories.get_tags_dir(path, split, label)
    logreg_dir = directories.get_logreg_dir(path, model_config, label)
    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')

    annotations = os.listdir(tags_dir) if os.path.exists(tags_dir) else None

    if not annotations:
        return

    features = [os.path.join(features_dir, x.replace('.json', '.npy')) for x in annotations]
    annotations = [os.path.join(tags_dir, x) for x in annotations]
    x = []
    y = []

    class_weight = {0: 0.5}

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
