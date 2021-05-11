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
from tools.sense_studio.project_utils import load_project_config
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


def train_logreg(path, split, label):
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    _, model_config = load_feature_extractor(path)

    features_dir = directories.get_features_dir(path, split, model_config, label=label)
    tags_dir = directories.get_tags_dir(path, split, label)
    logreg_dir = directories.get_logreg_dir(path, model_config, label)
    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')
    project_config = load_project_config(path)
    class_tags = project_config['classes'][label]

    if not os.path.exists(tags_dir):
        return

    annotation_files = os.listdir(tags_dir)

    feature_files = [os.path.join(features_dir, x.replace('.json', '.npy')) for x in annotation_files]
    annotation_files = [os.path.join(tags_dir, x) for x in annotation_files]
    all_features = []
    all_annotations = []

    for features in feature_files:
        features = np.load(features)

        for f in features:
            all_features.append(f.mean(axis=(1, 2)))

    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)['time_annotation']

        all_annotations.extend(annotations)

    # Reset tags that have been removed from the class to 'background'
    all_annotations = [tag_idx if tag_idx in class_tags else 0 for tag_idx in all_annotations]

    # Use low class weight for background and higher weight for all present tags
    annotated_tags = set(all_annotations)
    class_weight = {0: 0.5}
    class_weight.update({tag: 2 for tag in annotated_tags})

    all_features = np.array(all_features)
    all_annotations = np.array(all_annotations)

    if len(class_weight) > 1:
        logreg = LogisticRegression(C=0.1, class_weight=class_weight)
        logreg.fit(all_features, all_annotations)
        dump(logreg, logreg_path)
