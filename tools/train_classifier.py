#!/usr/bin/env python
"""
Finetuning script that can be used to train a custom classifier on top of our pretrained models.

Usage:
  train_classifier.py  --path_in=PATH
                       [--model_name=NAME]
                       [--model_version=VERSION]
                       [--num_layers_to_finetune=NUM]
                       [--epochs=NUM]
                       [--use_gpu]
                       [--path_out=PATH]
                       [--temporal_training]
                       [--resume]
                       [--overwrite]
  train_classifier.py  (-h | --help)

Options:
  --path_in=PATH                 Path to the dataset folder.
                                 Important: this folder should follow the structure described in the README.
  --model_name=NAME              Name of the backbone model to be used.
  --model_version=VERSION        Version of the backbone model to be used.
  --num_layers_to_finetune=NUM   Number of layers to finetune in addition to the final layer [default: 9].
  --epochs=NUM                   Number of epochs to run [default: 80].
  --path_out=PATH                Where to save results. Will default to `path_in` if not provided.
  --temporal_training            Use this flag if your dataset has been annotated with the temporal
                                 annotations tool
  --resume                       Initialize weights from the last saved checkpoint and restart training
  --overwrite                    Allow overwriting existing checkpoint files in the output folder (path_out)
"""
import datetime
import json
import os
import sys

from docopt import docopt
from natsort import natsorted
import torch.utils.data

from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.finetuning import extract_features
from sense.finetuning import generate_data_loader
from sense.finetuning import set_internal_padding_false
from sense.finetuning import training_loops
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig
from sense.loading import update_backbone_weights
from tools.sense_studio.project_utils import load_project_config
from sense.utils import clean_pipe_state_dict_key
from tools import directories


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', []),
    ModelConfig('StridedInflatedMobileNetV2', 'pro', []),
    ModelConfig('StridedInflatedEfficientNet', 'lite', []),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', []),
]


def train_model(path_in, path_out, model_name, model_version, num_layers_to_finetune, epochs,
                use_gpu=True, overwrite=True, temporal_training=None, resume=False, log_fn=print,
                confmat_event=None):
    os.makedirs(path_out, exist_ok=True)

    # Check for existing files
    saved_files = ["last_classifier.checkpoint", "best_classifier.checkpoint", "config.json", "label2int.json",
                   "confusion_matrix.png", "confusion_matrix.npy"]

    if not overwrite and any(os.path.exists(os.path.join(path_out, file)) for file in saved_files):
        print(f"Warning: This operation will overwrite files in {path_out}")

        while True:
            confirmation = input("Are you sure? Add --overwrite to hide this warning. (Y/N) ")
            if confirmation.lower() == "y":
                break
            elif confirmation.lower() == "n":
                sys.exit()
            else:
                print('Invalid input')

    # Load weights
    selected_config, weights = get_relevant_weights(
        SUPPORTED_MODEL_CONFIGURATIONS,
        model_name,
        model_version,
        log_fn,
    )
    backbone_weights = weights['backbone']

    if resume:
        # Load the last classifier
        checkpoint_classifier = torch.load(os.path.join(path_out, 'last_classifier.checkpoint'))

        # Update original weights in case some intermediate layers have been finetuned
        update_backbone_weights(backbone_weights, checkpoint_classifier)

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, backbone_weights)

    # Get the required temporal dimension of feature tensors in order to
    # finetune the provided number of layers
    if num_layers_to_finetune > 0:
        num_timesteps = backbone_network.num_required_frames_per_layer.get(-num_layers_to_finetune)
        if not num_timesteps:
            # Remove 1 because we added 0 to temporal_dependencies
            num_layers = len(backbone_network.num_required_frames_per_layer) - 1
            msg = (f'ERROR - Num of layers to finetune not compatible. '
                   f'Must be an integer between 0 and {num_layers}')
            log_fn(msg)
            raise IndexError(msg)
    else:
        num_timesteps = 1

    # Extract layers to finetune
    if num_layers_to_finetune > 0:
        fine_tuned_layers = backbone_network.cnn[-num_layers_to_finetune:]
        backbone_network.cnn = backbone_network.cnn[0:-num_layers_to_finetune]

    # finetune the model
    extract_features(path_in, selected_config, backbone_network, num_layers_to_finetune, use_gpu,
                     num_timesteps=num_timesteps, log_fn=log_fn)

    # Find label names
    label_names = os.listdir(directories.get_videos_dir(path_in, 'train'))
    label_names = natsorted(label_names)
    label_names = [x for x in label_names if not x.startswith('.')]
    label_names_temporal = ['background']

    project_config = load_project_config(path_in)
    if project_config:
        for temporal_tags in project_config['classes'].values():
            label_names_temporal.extend(temporal_tags)
    else:
        for label in label_names:
            label_names_temporal.extend([f'{label}_tag1', f'{label}_tag2'])

    label_names_temporal = sorted(set(label_names_temporal))

    label2int_temporal_annotation = {name: index for index, name in enumerate(label_names_temporal)}
    label2int = {name: index for index, name in enumerate(label_names)}

    extractor_stride = backbone_network.num_required_frames_per_layer_padding[0]

    # Create the data loaders
    features_dir = directories.get_features_dir(path_in, 'train', selected_config, num_layers_to_finetune)
    tags_dir = directories.get_tags_dir(path_in, 'train')
    train_loader = generate_data_loader(
        project_config,
        features_dir,
        tags_dir,
        label_names,
        label2int,
        label2int_temporal_annotation,
        num_timesteps=num_timesteps,
        stride=extractor_stride,
        temporal_annotation_only=temporal_training,
    )

    features_dir = directories.get_features_dir(path_in, 'valid', selected_config, num_layers_to_finetune)
    tags_dir = directories.get_tags_dir(path_in, 'valid')
    valid_loader = generate_data_loader(
        project_config,
        features_dir,
        tags_dir,
        label_names,
        label2int,
        label2int_temporal_annotation,
        num_timesteps=None,
        batch_size=1,
        shuffle=False,
        stride=extractor_stride,
        temporal_annotation_only=temporal_training,
    )

    # Check if the data is loaded fully
    if not train_loader or not valid_loader:
        log_fn("ERROR - \n "
               "\tMissing annotations for train or valid set.\n"
               "\tHint: Check if tags_train and tags_valid directories exist.\n")
        return

    # Modify the network to generate the training network on top of the features
    if temporal_training:
        num_output = len(label_names_temporal)
    else:
        num_output = len(label_names)

    # modify the network to generate the training network on top of the features
    gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                            num_out=num_output,
                                            use_softmax=False)

    if resume:
        gesture_classifier.load_state_dict(checkpoint_classifier)

    if num_layers_to_finetune > 0:
        # remove internal padding for training
        fine_tuned_layers.apply(set_internal_padding_false)
        net = Pipe(fine_tuned_layers, gesture_classifier)
    else:
        net = gesture_classifier
    net.train()

    if use_gpu:
        net = net.cuda()

    lr_schedule = {0: 0.0001, int(epochs / 2): 0.00001} if epochs > 1 else {0: 0.0001}
    num_epochs = epochs

    # Save training config and label2int dictionary
    config = {
        'backbone_name': selected_config.model_name,
        'backbone_version': selected_config.version,
        'num_layers_to_finetune': num_layers_to_finetune,
        'classifier': str(gesture_classifier),
        'temporal_training': temporal_training,
        'lr_schedule': lr_schedule,
        'num_epochs': num_epochs,
        'start_time': str(datetime.datetime.now()),
        'end_time': '',
    }
    with open(os.path.join(path_out, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(path_out, 'label2int.json'), 'w') as f:
        json.dump(label2int_temporal_annotation if temporal_training else label2int, f, indent=2)

    # Train model
    best_model_state_dict = training_loops(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule,
                                           label_names, label_names_temporal, path_out,
                                           temporal_annotation_training=temporal_training, log_fn=log_fn,
                                           confmat_event=confmat_event)

    # Save best model
    if isinstance(net, Pipe):
        best_model_state_dict = {clean_pipe_state_dict_key(key): value
                                 for key, value in best_model_state_dict.items()}
    torch.save(best_model_state_dict, os.path.join(path_out, "best_classifier.checkpoint"))

    config['end_time'] = str(datetime.datetime.now())
    with open(os.path.join(path_out, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    _path_in = args['--path_in']
    _path_out = args['--path_out'] or os.path.join(_path_in, "checkpoints")
    _use_gpu = args['--use_gpu']
    _model_name = args['--model_name'] or None
    _model_version = args['--model_version'] or None
    _num_layers_to_finetune = int(args['--num_layers_to_finetune'])
    _epochs = int(args['--epochs'])
    _temporal_training = args['--temporal_training']
    _resume = args['--resume']
    _overwrite = args['--overwrite']

    train_model(
        path_in=_path_in,
        path_out=_path_out,
        model_name=_model_name,
        model_version=_model_version,
        num_layers_to_finetune=_num_layers_to_finetune,
        epochs=_epochs,
        use_gpu=_use_gpu,
        overwrite=_overwrite,
        temporal_training=_temporal_training,
        resume=_resume,
    )
