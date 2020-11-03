#!/usr/bin/env python
"""
Finetuning script that can be used to train a custom classifier on top of our pretrained models.

Usage:
  train_classifier.py  --path_in=PATH
                       [--num_layers_to_finetune=NUM]
                       [--use_gpu]
  train_classifier.py  (-h | --help)

Options:
  --path_in=PATH                Path to the dataset folder.
                                Important: this folder should follow the structure described in the README.
  --num_layers_to_finetune=NUM  Number layer to finetune, must be integer between 0 and 32 [default: 9]

"""

from docopt import docopt

import json
import os

import torch.utils.data

from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet.finetuning import training_loops, extract_features, generate_data_loader
from realtimenet.finetuning import set_internal_padding_false
from realtimenet import feature_extractors


num_layers2timesteps = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 3,
    8: 3,
    9: 5,
    10: 5,
    11: 5,
    12: 7,
    13: 7,
    14: 7,
    15: 9,
    16: 9,
    17: 9,
    18: 19,
    19: 19,
    20: 19,
    21: 21,
    22: 21,
    23: 21,
    24: 21,
    25: 43,
    26: 43,
    27: 43,
    28: 43,
    29: 45,
    30: 45,
    31: 45,
    32: 45
}
MIN_CLIP_TIMESTEP = 45


def clean_pipe_state_dict_key(key):
    to_remove = [
        'feature_extractor.',
        'feature_converter.'
    ]
    for pattern in to_remove:
        if key.startswith(pattern):
            key = key.replace(pattern, '')
    return key


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    path_in = args['--path_in']
    use_gpu = args['--use_gpu']
    num_layers_to_finetune = int(args['--num_layers_to_finetune'])

    # compute the number of timestep necessary for each video features in order to finetune the number of layer wished.
    num_timestep = num_layers2timesteps.get(int(num_layers_to_finetune))
    if not num_timestep:
        raise NameError('Num layers to finetune not right. Must be integer between 0 and 32.')

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    # remove internal padding for feature extraction and training
    feature_extractor.apply(set_internal_padding_false)
    checkpoint = torch.load('resources/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Concatenate feature extractor and met converter
    if num_layers_to_finetune > 0:
        custom_classifier_bottom = feature_extractor.cnn[-num_layers_to_finetune:]
        feature_extractor.cnn = feature_extractor.cnn[0:-num_layers_to_finetune]

    # list the labels from the training directory
    videos_dir = os.path.join(path_in, "videos_train")
    features_dir = os.path.join(path_in, "features_train")
    classes = os.listdir(videos_dir)
    classes = [x for x in classes if not x.startswith('.')]

    # finetune the model
    extract_features(path_in, feature_extractor, num_layers_to_finetune, use_gpu,
                     minimum_frames=MIN_CLIP_TIMESTEP)

    class2int = {x: e for e, x in enumerate(classes)}

    # create the data loaders
    train_loader = generate_data_loader(os.path.join(path_in, f"features_train_{num_layers_to_finetune}"),
                                        classes, class2int, num_timesteps=num_timestep)
    valid_loader = generate_data_loader(os.path.join(path_in, f"features_valid_{num_layers_to_finetune}"),
                                        classes, class2int, num_timesteps=None, batch_size=1, shuffle=False)


    # modeify the network to generate the training network on top of the features
    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=len(classes))
    if num_layers_to_finetune > 0:
        net = Pipe(custom_classifier_bottom, gesture_classifier)
    else:
        net = gesture_classifier
    net.train()

    if use_gpu:
        net = net.cuda()

    lr_schedule = {0: 0.0001, 40: 0.00001}
    num_epochs = 60
    best_model_state_dict = training_loops(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule)

    # Save best model
    if isinstance(net, Pipe):
        best_model_state_dict = {clean_pipe_state_dict_key(key): value
                                 for key, value in best_model_state_dict.items()}
    torch.save(best_model_state_dict, os.path.join(path_in, "classifier.checkpoint"))
    json.dump(class2int, open(os.path.join(path_in, "class2int.json"), "w"))
