import json
import os
import torch
import yaml

from typing import List
from typing import Optional
from typing import Tuple

from sense import RESOURCES_DIR
from sense import backbone_networks

with open(os.path.join(os.getcwd(), os.path.dirname(__file__), 'models.yml')) as f:
    MODELS = yaml.load(f, Loader=yaml.FullLoader)


class ModelConfig:
    """
    Object containing the model specifications for downstream tasks.

    The full list of available models can be found in `sense/models.yml`
    """

    def __init__(self, model_name: str, version: str, feature_converters: List[str]):
        """
        :param model_name:
            Name of the model to use (StridedInflatedEfficientNet or StridedInflatedMobileNetV2)
        :param version:
            Model version to use (pro or lite)
        :param feature_converters:
            List of classifier heads on top of the backbone network
        """

        all_model_names = sorted(MODELS.keys())
        if model_name not in all_model_names:
            raise Exception(f'Unknown model name: {model_name}. '
                            f'\nAvailable models: {all_model_names}')

        all_versions = sorted(MODELS[model_name].keys())
        if version not in all_versions:
            raise Exception(f'Version {version} is not available for this model (={model_name}).'
                            f'\nAvailable versions: {all_versions}')

        all_feature_converters = sorted(MODELS[model_name][version].keys())
        for feature_converter in feature_converters:
            if feature_converter not in all_feature_converters:
                raise Exception(f'The {version} version of {model_name} does not support '
                                f'{feature_converter} as a downstream task.'
                                f'\nAvailable converters: {all_feature_converters}')

        self.model_name = model_name
        self.version = version
        self.feature_converters = feature_converters

    def get_path_weights(self):
        model_weights = MODELS[self.model_name][self.version]
        return {name: model_weights[name] for name in ['backbone'] + self.feature_converters}


def get_relevant_weights(model_config_list: List[ModelConfig], requested_model_name=None,
                         requested_version=None) -> Optional[Tuple[ModelConfig, dict]]:
    """
    Returns the model weights for the appropriate backbone and classifier head based on
    a list of compatible model configs. The first available config is returned.

    :param model_config_list:
        List of compatible model configurations
    :param requested_model_name:
        Name of a specific model to use (i.e. StridedInflatedEfficientNet or StridedInflatedMobileNetV2)
    :param requested_version:
        Version of the model to use (i.e. pro or lite)
    :return:
        First available model config and dictionary of model weights
    """

    # Filter out model configurations that don't match requested name and version
    if requested_model_name:
        model_config_list = [config for config in model_config_list
                             if config.model_name == requested_model_name]
    if requested_version:
        model_config_list = [config for config in model_config_list
                             if config.version == requested_version]

    # Check if not empty
    if not model_config_list:
        raise Exception(f'Could not find a model configuration matching requested parameters:\n'
                        f'\tmodel_name={requested_model_name}\n'
                        f'\tversion={requested_version}')

    for model_config in model_config_list:
        path_weights = model_config.get_path_weights()
        path_weights_string = json.dumps(path_weights, indent=4, sort_keys=True)  # used in prints

        if all(os.path.exists(path) for path in path_weights.values()) or using_travis():
            print(f'Weights found:\n{path_weights_string}')
            weights = {}
            for name, path in path_weights.items():
                load_fn = load_backbone_weights if name == 'backbone' else load_weights
                weights[name] = load_fn(path)

            return model_config, weights
        else:
            print(f'Could not find at least one of the following files:\n{path_weights_string}')

    raise Exception('ERROR - Weights file missing. To download, please go to '
                    'https://20bn.com/licensing/sdk/evaluation and follow the '
                    'instructions.')


def load_weights(checkpoint_path: str):
    """
    Load weights from a checkpoint file.

    :param checkpoint_path:
        A string representing the absolute/relative path to the checkpoint file.
    """
    return torch.load(checkpoint_path, map_location='cpu')


def load_weights_from_resources(checkpoint_path: str):
    """
    Load weights from a checkpoint file located in the resources folder.

    :param checkpoint_path:
        A string representing the absolute/relative path to the checkpoint file.
    """
    checkpoint_path = os.path.join(RESOURCES_DIR, checkpoint_path.split(f'resources{os.sep}')[-1])
    try:
        return load_weights(checkpoint_path)

    except FileNotFoundError:
        raise FileNotFoundError('Weights file missing: {}. '
                                'To download, please go to '
                                'https://20bn.com/licensing/sdk/evaluation and follow the '
                                'instructions.'.format(checkpoint_path))


def load_backbone_weights(checkpoint_path: str):
    """
    Load backbone weights from a checkpoint file, unless Travis is used. Raises an error pointing
    to the SDK page in case weights are missing.

    :param checkpoint_path:
        A string representing the absolute/relative path to the checkpoint file.
    """
    if not using_travis():
        return load_weights_from_resources(checkpoint_path)
    else:
        print('Weights are not loaded on Travis.')


def build_backbone_network(selected_config: ModelConfig, weights: dict):
    """
    Creates a backbone network and load provided weights, unless Travis is used.

    :param selected_config:
        An instance of ModelConfig, specifying the backbone architecture name.
    :param weights:
        A model state dict.
    :return:
        A backbone network, with pre-trained weights.
    """
    backbone_network = getattr(backbone_networks, selected_config.model_name)()
    if not using_travis():
        backbone_network.load_state_dict(weights)
    backbone_network.eval()
    return backbone_network


def using_travis():
    """
    Returns True if Travis is currently being used.
    """
    return os.getenv('TRAVIS', False) == 'true'
