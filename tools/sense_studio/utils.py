from sense.engine import InferenceEngine
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig

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
