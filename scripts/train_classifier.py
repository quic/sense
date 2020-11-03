#!/usr/bin/env python
"""
Real time detection of 30 hand gestures.

Usage:
  train_classifier.py  --path_in=PATH
                       [--num_layer_finetune=NUM]
                       [--use_gpu]
  train_classifier.py  (-h | --help)

Options:
  --path_in=PATH            path to the dataset folder following the structure described in the readme
  --num_layer_finetune=NUM  Number layer to finetune, must be integer between 0 and 32 [default: 9]

"""

import torch.utils.data
from docopt import docopt
import os

from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet.finetune_utils import training_loops, extract_features, generate_data_loader, evaluation_model
from realtimenet.finetune_utils import set_internal_padding_false
from realtimenet import feature_extractors
import json


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

if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    path_in = args['--path_in']
    use_gpu = args['--use_gpu']
    num_layer_finetune = int(args['--num_layer_finetune'])

    # compute the number of timestep necessary for each video features in order to finetune the number of layer wished.
    num_timestep = num_layers2timesteps.get(int(num_layer_finetune))
    if not num_timestep:
        raise NameError('Num layers to finetune not right. Must be integer between 0 and 32.')

    lr_schedule = {0: 0.0001, 40: 0.00001}
    num_epochs = 60

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    # remove internal padding for feature extraction and training
    feature_extractor.apply(set_internal_padding_false)
    checkpoint = torch.load('resources/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Concatenate feature extractor and met converter
    if num_layer_finetune > 0:
        custom_classifier_bottom = feature_extractor.cnn[-num_layer_finetune:]
        feature_extractor.cnn = feature_extractor.cnn[0:-num_layer_finetune]

    # list the labels from the training directory
    videos_dir = os.path.join(path_in, "videos_train")
    features_dir = os.path.join(path_in, "features_train")
    classes = os.listdir(videos_dir)
    classes = [x for x in classes if not x.startswith('.')]

    # finetune the model
    extract_features(path_in, classes, feature_extractor, num_layer_finetune, use_gpu, minimum_frames=MIN_CLIP_TIMESTEP)

    class2int = {x: e for e, x in enumerate(classes)}

    # create the data loaders
    trainloader = generate_data_loader(os.path.join(path_in, "features_train_" + str(num_layer_finetune)),
                                       classes, class2int, num_timesteps=num_timestep)


    # modeify the network to generate the training network on top of the features
    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=len(classes))
    if num_layer_finetune > 0:
        net = Pipe(custom_classifier_bottom, gesture_classifier)
    else:
        net = gesture_classifier
    net.train()

    if use_gpu:
        net = net.cuda()

    valid_features_dir = os.path.join(path_in, "features_valid_" + str(num_layer_finetune))

    training_loops(net, trainloader, use_gpu, num_epochs, lr_schedule, valid_features_dir,
                   classes, class2int, num_timestep)

    # save the trained model
    if num_layer_finetune > 0:
        state_dict = {**net.feature_extractor.state_dict(), **net.feature_converter.state_dict()}
    else:
        state_dict = net.state_dict()

    torch.save(state_dict, os.path.join(path_in, "classifier.checkpoint"))
    json.dump(class2int, open(os.path.join(path_in, "class2int.json"), "w"))

