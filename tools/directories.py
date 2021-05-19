import os

from typing import List
from typing import Optional

from sense.loading import ModelConfig


def _get_data_dir(dir_type: str, dataset_path: str, split: Optional[str] = None, subdirs: Optional[List[str]] = None):
    main_dir = f'{dir_type}_{split}' if split else dir_type
    subdirs = subdirs or []

    return os.path.join(dataset_path, main_dir, *subdirs)


def get_videos_dir(dataset_path, split, label=None):
    subdirs = [label] if label else None
    return _get_data_dir('videos', dataset_path, split, subdirs)


def get_frames_dir(dataset_path, split, label=None):
    subdirs = [label] if label else None
    return _get_data_dir('frames', dataset_path, split, subdirs)


def get_features_dir(dataset_path, split, model: Optional[ModelConfig] = None, num_layers_to_finetune=0, label=None):
    subdirs = None
    if model:
        subdirs = [model.combined_model_name, f'num_layers_to_finetune={num_layers_to_finetune}']
        if label:
            subdirs.append(label)

    return _get_data_dir('features', dataset_path, split, subdirs)


def get_tags_dir(dataset_path, split, label=None):
    subdirs = [label] if label else None
    return _get_data_dir('tags', dataset_path, split, subdirs)


def get_logreg_dir(dataset_path, model: Optional[ModelConfig] = None):
    subdirs = None
    if model:
        subdirs = [model.combined_model_name]

    return _get_data_dir('logreg', dataset_path, subdirs=subdirs)
