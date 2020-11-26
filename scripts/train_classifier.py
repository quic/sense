#!/usr/bin/env python
"""
Finetuning script that can be used to train a custom classifier on top of our pretrained models.

Usage:
  train_classifier.py  --path_in=PATH
                       [--num_layers_to_finetune=NUM]
                       [--use_gpu]
                       [--path_out=PATH]
                       [--path_annotations_train=PATH]
                       [--path_annotations_valid=PATH]
  train_classifier.py  (-h | --help)

Options:
  --path_in=PATH                 Path to the dataset folder.
                                 Important: this folder should follow the structure described in the README.
  --num_layers_to_finetune=NUM   Number of layers to finetune in addition to the final layer [default: 9].
  --path_out=PATH                Where to save results. Will default to `path_in` if not provided.
  --path_annotations_train=PATH  Path to an annotation file. This argument is only useful if you want
                                 to fit a subset of the available training data. If provided, each entry
                                 in the json file should have the following format: {'file': NAME,
                                 'label': LABEL}.
  --path_annotations_valid=PATH  Same as '--path_annotations_train' but for validation examples.
"""

from docopt import docopt

import json
import os

import torch.utils.data

from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet.finetuning import training_loops, extract_features, generate_data_loader
from realtimenet.finetuning import set_internal_padding_false
from realtimenet import feature_extractors


def clean_pipe_state_dict_key(key):
    to_replace = [
        ('feature_extractor', 'cnn'),
        ('feature_converter.', '')
    ]
    for pattern, replacement in to_replace:
        if key.startswith(pattern):
            key = key.replace(pattern, replacement)
    return key


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    path_in = args['--path_in']
    path_out = args['--path_out'] or path_in
    os.makedirs(path_out, exist_ok=True)
    use_gpu = args['--use_gpu']
    path_annotations_train = args['--path_annotations_train'] or None
    path_annotations_valid = args['--path_annotations_valid'] or None
    num_layers_to_finetune = int(args['--num_layers_to_finetune'])

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    # remove internal padding for feature extraction and training
    feature_extractor.apply(set_internal_padding_false)
    checkpoint = torch.load('resources/backbone/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Get the require temporal dimension of feature tensors in order to
    # finetune the provided number of layers.
    if num_layers_to_finetune > 0:
        num_timesteps = feature_extractor.num_required_frames_per_layer.get(-num_layers_to_finetune)
        if not num_timesteps:
            num_layers = len(feature_extractor.num_required_frames_per_layer) - 1  # remove 1 because we
                                                                           # added 0 to temporal_dependencies
            raise IndexError(f'Num of layers to finetune not compatible. '
                             f'Must be an integer between 0 and {num_layers}')
    else:
        num_timesteps = 1
    minimum_frames = feature_extractor.num_required_frames_per_layer[0]

    # Concatenate feature extractor and met converter
    if num_layers_to_finetune > 0:
        custom_classifier_bottom = feature_extractor.cnn[-num_layers_to_finetune:]
        feature_extractor.cnn = feature_extractor.cnn[0:-num_layers_to_finetune]

    # finetune the model
    extract_features(path_in, feature_extractor, num_layers_to_finetune, use_gpu,
                     minimum_frames=minimum_frames)

    # Find label names
    label_names = os.listdir(os.path.join(os.path.join(path_in, "videos_train")))
    label_names = [x for x in label_names if not x.startswith('.')]
    label_counting = ['counting_background']
    for label in label_names:
        label_counting += [f'{label}_position_1', f'{label}_position_2']
    label2int_temporal_annotation = {name: index for index, name in enumerate(label_counting)}
    label2int = {name: index for index, name in enumerate(label_names)}

    extractor_stride = feature_extractor.num_required_frames_per_layer_padding[0]

    # create the data loaders
    train_loader = generate_data_loader(path_in, f"features_train_num_layers_to_finetune={num_layers_to_finetune}", "tags_train",
                                        label_names, label2int, label2int_temporal_annotation,  model_time_step=num_timesteps,
                                        num_timesteps=num_timesteps, minimum_frames=minimum_frames, stride=extractor_stride,
                                        temporal_annotation_only=False)

    valid_loader = generate_data_loader(path_in, f"features_valid_num_layers_to_finetune={num_layers_to_finetune}", "tags_valid",
                                        label_names, label2int, label2int_temporal_annotation, model_time_step=num_timesteps,
                                                 num_timesteps=None, batch_size=1, shuffle=False, minimum_frames=minimum_frames, stride=extractor_stride,
                                        temporal_annotation_only=False)


    # modeify the network to generate the training network on top of the features
    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=len(label_names))
    if num_layers_to_finetune > 0:
        net = Pipe(custom_classifier_bottom, gesture_classifier)
    else:
        net = gesture_classifier
    net.train()

    if use_gpu:
        net = net.cuda()

    lr_schedule = {0: 0.0001, 40: 0.00001}
    num_epochs = 80
    best_model_state_dict = training_loops(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule, label_names, path_out)

    # Save best model
    if isinstance(net, Pipe):
        best_model_state_dict = {clean_pipe_state_dict_key(key): value
                                 for key, value in best_model_state_dict.items()}
    torch.save(best_model_state_dict, os.path.join(path_out, "classifier.checkpoint"))
    json.dump(label2int, open(os.path.join(path_out, "label2int.json"), "w"))
