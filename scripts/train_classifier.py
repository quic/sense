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
                       [--temporal_training]
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
import json
import os
import torch.utils.data

from docopt import docopt

from sense import feature_extractors
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.finetuning import extract_features
from sense.finetuning import generate_data_loader
from sense.finetuning import set_internal_padding_false
from sense.finetuning import training_loops


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
    temporal_training = args['--temporal_training']
    if not temporal_training:
        temporal_training = False

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
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
        fine_tuned_layers = feature_extractor.cnn[-num_layers_to_finetune:]
        feature_extractor.cnn = feature_extractor.cnn[0:-num_layers_to_finetune]

    # finetune the model
    extract_features(path_in, feature_extractor, num_layers_to_finetune, use_gpu,
                     num_timesteps=num_timesteps)

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
                                        label_names, label2int, label2int_temporal_annotation,
                                        num_timesteps=num_timesteps, minimum_frames=minimum_frames, stride=extractor_stride,
                                        temporal_annotation_only=temporal_training)

    valid_loader = generate_data_loader(path_in, f"features_valid_num_layers_to_finetune={num_layers_to_finetune}", "tags_valid",
                                        label_names, label2int, label2int_temporal_annotation,
                                        num_timesteps=None, batch_size=1, shuffle=False, minimum_frames=minimum_frames, stride=extractor_stride,
                                        temporal_annotation_only=temporal_training)

    # modeify the network to generate the training network on top of the features
    if temporal_training:
        num_output = len(label_counting)
    else:
        num_output = len(label_names)

    # remove internal padding for training
    if num_layers_to_finetune > 0:
        fine_tuned_layers.apply(set_internal_padding_false)
    # modify the network to generate the training network on top of the features
    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=num_output,
                                            use_softmax=False)

    if num_layers_to_finetune > 0:
        net = Pipe(fine_tuned_layers, gesture_classifier)
    else:
        net = gesture_classifier
    net.train()

    if use_gpu:
        net = net.cuda()

    lr_schedule = {0: 0.0001, 40: 0.00001}
    num_epochs = 80
    best_model_state_dict = training_loops(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule, label_names, path_out, temporal_annotation_training=temporal_training)

    # Save best model
    if isinstance(net, Pipe):
        best_model_state_dict = {clean_pipe_state_dict_key(key): value
                                 for key, value in best_model_state_dict.items()}
    torch.save(best_model_state_dict, os.path.join(path_out, "classifier.checkpoint"))
    if temporal_training:
        json.dump(label2int_temporal_annotation, open(os.path.join(path_out, "label2int.json"), "w"))
    else:
        json.dump(label2int, open(os.path.join(path_out, "label2int.json"), "w"))
