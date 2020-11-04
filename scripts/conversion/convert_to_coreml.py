#! /usr/bin/env python
"""
Pytorch-to-CoreML conversion script.
Author: Mark Tordorovich.


Usage:
  convert_to_coreml.py --backbone=NAME --classifier=NAME --output_name=NAME
                       [--path_in=PATH]
                       [--plot_model]
                       [--float32]
  convert_to_coreml.py (-h | --help)

Options:
  --backbone=NAME     Name of the backbone model.
  --classifier=NAME   Name of the classifier model.
  --output_name=NAME  Base name of the output file.
  --path_in=PATH      Path to the trained classifier directory if converting custom classifier. [default: None]
  --plot_model        Plot intermediate Keras model and save as image.
  --float32           Use full precision. By default, this script quantizes weights
                      to 16-bit precision.
  -h --help
"""

from docopt import docopt
import configparser
import io
import json
import os
import copy
from collections import defaultdict

import torch
import numpy as np

from keras import backend as K
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add, Dense, GlobalAveragePooling2D,
                          UpSampling2D, MaxPooling2D, Concatenate, DepthwiseConv2D, Softmax)
from keras.layers.advanced_activations import (LeakyReLU, ReLU, PReLU)
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomNormal
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot
import coremltools


DEFAULT_CONVERSION_PARAMETERS = {
    'image_scale': 1.,
    'normalize_inputs': False,
    'red_bias': None,
    'green_bias': None,
    'blue_bias': None,
    'red_scale': None,
    'green_scale': None,
    'blue_scale': None,
    'use_prelu': False
}

SUPPORTED_BACKBONE_CONVERSIONS = {
    'efficientnet':
        {
            'config_file': 'scripts/conversion/cfg/efficientnet.cfg',
            'weights_file': 'resources/backbone/strided_inflated_efficientnet.ckpt',
            'conversion_parameters': {**DEFAULT_CONVERSION_PARAMETERS, 'image_scale': 255.}
        }
}

SUPPORTED_CLASSIFIER_CONVERSIONS = {
    'efficient_net_gesture_control':
        {
            'config_file': 'scripts/conversion/cfg/logistic_regression.cfg',
            'placeholder_values': {'NUM_CLASSES': '30'},
            'weights_file': 'resources/gesture_detection/efficientnet_logistic_regression.ckpt',
            'corresponding_backbone': 'efficientnet',
        },
    'custom_classifier':
        {
            'config_file': 'scripts/conversion/cfg/logistic_regression.cfg',
            'placeholder_values': {'NUM_CLASSES': None},
            'weights_file': None,
            'corresponding_backbone': None,
        }
}


def merge_backbone_and_classifier_cfg_files(backbone_config_file, classifier_config_file,
                                            placeholder_values=None):
    """
    Concatenate backbone and classifier config files and make sure all config sections
    have unique names (adding unique suffixes) for compatibility with configparser.
    """
    placeholder_values = placeholder_values or {}
    section_counters = defaultdict(int)
    output_stream = io.StringIO()

    for cfg_file in [backbone_config_file, classifier_config_file]:

        with open(cfg_file) as fin:
            for line in fin:
                # Make sure section names are unique
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)

                for key, value in placeholder_values.items():
                    line = line.replace(key, value)

                output_stream.write(line)

    output_stream.seek(0)
    return output_stream


def convert(backbone_settings, classifier_settings, output_name, float32, plot_model):
    output_dir = 'resources/coreml/'
    os.makedirs(output_dir, exist_ok=True)

    conversion_parameters = backbone_settings['conversion_parameters']
    keras_file = os.path.join(output_dir, output_name + '.h5')
    coreml_file = os.path.join(output_dir, output_name + '.mlmodel')

    if plot_model:
        plot_file = os.path.join(output_dir, output_name + '.png')

    # Load weights and config.
    print('Loading weights.')
    weights_backbone = torch.load(backbone_settings['weights_file'],
                                  map_location='cpu')
    weights_classifier = torch.load(classifier_settings['weights_file'],
                                    map_location='cpu')
    # if some deeper layer have been finetuned, change them in the backbone weights dictionary
    name_finetuned_layers = set(weights_backbone.keys()).intersection(weights_classifier.keys())
    for key in name_finetuned_layers:
        weights_backbone[key] = weights_classifier.pop(key)
    weights_full = {**weights_backbone, **weights_classifier}

    for key in weights_full.keys():
        print(key, weights_full[key].shape)

    print('Parsing CFG file.')
    placeholder_values= {**backbone_settings.get('placeholder_values', {}),
                         **classifier_settings.get('placeholder_values', {})}
    unique_config_file = merge_backbone_and_classifier_cfg_files(backbone_settings['config_file'],
                                                                 classifier_settings['config_file'],
                                                                 placeholder_values=placeholder_values)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    weight_decay = 5e-4

    def invResidual(module_name, layer_name, frames, out_channels, xratio, size, stride, shift,
                    tstride, fake_weights):

        s = 0
        if shift:
            print('3D conv block')
            tsize = 3
        else:
            print('2D conv block')
            tsize = 1
        prev_layer_shape = K.int_shape(all_layers[-1])
        input_channels = prev_layer_shape[-1]
        x_channels = input_channels * xratio
        image_size = prev_layer_shape[-3], prev_layer_shape[-2]
        print('input image size: ', image_size)
        num_convs = int(frames / tstride)
        inputs_needed = (tstride * (num_convs - 1)) + tsize
        #            inputs_needed = frames + tsize - 1
        if inputs_needed > 1:
            print('inputs_needed: ', inputs_needed)
        old_frames_to_read = inputs_needed - frames
        new_frames_to_save = min(frames, old_frames_to_read)
        print('num_convs: ', num_convs,
              'inputs_needed: ', inputs_needed,
              'history frames needed: ', old_frames_to_read,
              'frames to save: ', new_frames_to_save,
              'tstride: ', tstride)
        # create (optional) expansion pointwise convolution layer

        input_indexes = []
        for i in range(num_convs):
            input_indexes.append(len(all_layers) - frames + (i * tstride))

        if xratio != 1:
            print('---------- Insert channel multiplier pointwise conv -------------')
            # attach output ports to inputs we will need next pass if tsize>1
            for f in range(new_frames_to_save):
                out_index.append(len(all_layers) - frames + f)
                out_names.append(module_name + '_save_' + str(f))

            # create input ports for required old frames if tsize>1
            for f in range(old_frames_to_read):
                h_name = module_name + '_history_' + str(f)
                all_layers.append(
                    Input(shape=(image_size[0], image_size[1], input_channels), name=h_name))
                in_names.append(h_name)
                in_index.append(len(all_layers) - 1)

            # get weights
            n = module_name + '.conv.' + str(s) + '.0.'
            if n + 'weight' in weights_full:
                weights_pt = weights_full[n + 'weight']
                print('checkpoint: ', weights_pt.shape, )
                weights_k = np.transpose(weights_pt, [2, 3, 1, 0])
                bias = weights_full[n + 'bias']
            else:
                print('missing weight ', n + 'weight')
                weights_k = np.random.rand(1, 1, tsize * input_channels, x_channels)
                bias = np.zeros(x_channels)
                fake_weights = True

            expected_weights_shape = (1, 1, tsize * input_channels, x_channels)
            print('weight shape, expected : ', expected_weights_shape,
                  'transposed: ', weights_k.shape)

            if (weights_k.shape != expected_weights_shape):
                print('weight matrix shape is wrong, making a fake one')
                weights_k = np.random.rand(1, 1, tsize * input_channels, x_channels)
                bias = np.zeros(x_channels)
                fake_weights = True

            weights = [weights_k, bias]

            inputs = []
            outputs = []

            for f in range(inputs_needed):
                #                print('input index: ', len(all_layers) - inputs_needed + f, ' shape: ',
                #                      all_layers[len(all_layers) - inputs_needed + f].shape)
                inputs.append(all_layers[len(all_layers) - inputs_needed + f])
                if merge_in > 0:
                    #                    print('merge input index: ', len(all_layers) - inputs_needed + f, ' shape: ',
                    #                          all_layers[len(all_layers) - (2 * inputs_needed) + f].shape)
                    inputs.append(all_layers[len(all_layers) - (2 * inputs_needed) + f])

            for f in range(int(frames / tstride)):
                layers = []
                if tsize > 1:
                    #                    print('concatenate layers:')
                    for t in range(tsize):
                        # offset is constant with f, except if tstride,
                        # then steps by extra step every time through
                        #                        layers.append(inputs[t + (f * (tstride))])
                        layers.append(inputs[(tsize - t - 1) + (f * (tstride))])
                    cat_layer = Concatenate()(layers)
                else:
                    cat_layer = inputs[f * (tstride)]

                outputs.append((Conv2D(
                    x_channels, (1, 1),
                    use_bias=not batch_normalize,
                    weights=weights,
                    activation=None,
                    padding='same'))(cat_layer))

            print('parallel convs: ', int(frames / tstride), ' : ', K.int_shape(cat_layer))

            if activation == 'leaky':
                for f in range(int(frames / tstride)):
                    if not conversion_parameters['use_prelu']:
                        outputs[f] = LeakyReLU(alpha=0.1)(outputs[f])
                    else:
                        outputs[f] = PReLU(
                            alpha_initializer=RandomNormal(mean=0.1, stddev=0.0, seed=None),
                            shared_axes=[1, 2])(outputs[f])
            elif activation == 'relu6':
                for f in range(int(frames / tstride)):
                    outputs[f] = ReLU(max_value=6)(outputs[f])

            for f in range(int(frames / tstride)):
                all_layers.append(outputs[f])
            s += 1
            frames = int(frames / tstride)

        else:
            print('Skipping channel multiplier pointwise conv, no expansion')

        # create groupwise convolution
        # get weights
        print('---------- Depthwise conv -------------')
        n = module_name + '.conv.' + str(s) + '.0.'
        print('module name base: ', n)
        if n + 'weight' in weights_full:
            weights_pt = weights_full[n + 'weight']
            print('checkpoint: ', weights_pt.shape, )
            weights_k = np.transpose(weights_pt, [2, 3, 0, 1])
            bias = weights_full[n + 'bias']
        else:
            print('missing weight ', n + 'weight')
            weights_k = np.random.rand(size, size, x_channels, 1)
            bias = np.zeros(x_channels)
            fake_weights = True

        expected_weights_shape = (size, size, x_channels, 1)
        print('weight shape, expected : ', expected_weights_shape,
              'transposed: ', weights_k.shape)

        if (weights_k.shape != expected_weights_shape):
            print('weight matrix shape is wrong, making a fake one')
            fake_weights = True
            weights_k = np.random.rand(size, size, x_channels, 1)
            bias = np.zeros(x_channels)

        weights = [weights_k, bias]

        inputs = []
        outputs = []

        padding = 'same' if pad == 1 and stride == 1 else 'valid'

        for f in range(frames):
            #            print('input index: ', len(all_layers) - frames + f, ' shape: ',
            #                  all_layers[len(all_layers) - frames + f].shape)
            inputs.append(all_layers[len(all_layers) - frames + f])

        if stride > 1:
            for f in range(len(inputs)):
                if size == 3:  # originally for all sizes
                    inputs[f] = ZeroPadding2D(((size - stride, 0), (size - stride, 0)))(inputs[f])
                elif size == 5:  # I found this works...
                    inputs[f] = ZeroPadding2D(((2, 2), (2, 2)))(inputs[f])
                else:
                    print('I have no idea what to do for size ', size)
                    exit()

        print('parallel convs: ', f, ' : ', K.int_shape(inputs[0]), 'padding: ', padding)
        for f in range(frames):
            outputs.append((DepthwiseConv2D(
                (size, size),
                strides=(stride, stride),
                use_bias=not batch_normalize,
                weights=weights,
                activation=None,
                padding=padding))(inputs[f]))

        if activation == 'leaky':
            for f in range(int(frames)):
                if not conversion_parameters['use_prelu']:
                    outputs[f] = LeakyReLU(alpha=0.1)(outputs[f])
                else:
                    outputs[f] = PReLU(
                        alpha_initializer=RandomNormal(mean=0.1, stddev=0.0, seed=None),
                        shared_axes=[1, 2])(outputs[f])
        elif activation == 'relu6':
            for f in range(int(frames)):
                outputs[f] = ReLU(max_value=6)(outputs[f])

        for f in range(int(frames)):
            all_layers.append(outputs[f])
        s += 1

        # create pointwise convolution
        # get weights
        print('---------- Pointwise conv -------------')
        n = module_name + '.conv.' + str(s) + '.'
        print('module name base: ', n)
        if n + 'weight' in weights_full:
            weights_pt = weights_full[n + 'weight']
            print('checkpoint: ', weights_pt.shape, )
            weights_k = np.transpose(weights_pt, [2, 3, 1, 0])
            bias = weights_full[n + 'bias']
        else:
            print('missing weight ', n + 'weight')
            fake_weights = True
            weights_k = np.random.rand(1, 1, x_channels, out_channels)
            bias = np.zeros(out_channels)

        expected_weights_shape = (1, 1, x_channels, out_channels)
        print('weight shape, expected : ', expected_weights_shape,
              'transposed: ', weights_k.shape)

        if (weights_k.shape != expected_weights_shape):
            print('weight matrix shape is wrong, making a fake one')
            fake_weights = True
            weights_k = np.random.rand(1, 1, x_channels, out_channels)
            bias = np.zeros(out_channels)

        weights = [weights_k, bias]
        print("combined shape: ", weights[0].shape, weights[1].shape)

        inputs = []
        outputs = []

        for f in range(frames):
            #            print('input index: ', len(all_layers) - frames + f, ' shape: ',
            #                  all_layers[len(all_layers) - frames + f].shape)
            inputs.append(all_layers[len(all_layers) - frames + f])

        print('parallel convs: ', f, ' : ', K.int_shape(all_layers[len(all_layers) - frames]))
        for f in range(frames):
            conv_input = all_layers[len(all_layers) - frames + f]

            outputs.append((Conv2D(
                out_channels, (1, 1),
                use_bias=not batch_normalize,
                weights=weights,
                activation=None,
                padding='same'))(conv_input))

        if stride == 1 and input_channels == out_channels:
            for f in range(int(frames)):
                #                if tstride==2:
                #                    out_index.append(input_indexes[f])
                #                    out_names.append('outputx_' + str(f) + layer_name)

                all_layers.append(Add()([all_layers[input_indexes[f]], outputs[f]]))
        else:
            for f in range(int(frames)):
                all_layers.append(outputs[f])
        s += 1

        return frames, fake_weights

    print('Creating Keras model.')
    all_layers = []
    out_index = []
    out_names = []
    in_index = []
    in_names = []
    image_inputs = []
    layer_names = dict()
    frames = 0
    coreml_list = []
    fake_weights = False
    np.random.seed(13)  # start the same way each time...

    for section in cfg_parser.sections():
        print('    ***** Parsing section {} ************'.format(section))
        if section.startswith('convolutional'):
            if frames > 1:
                print('frames: ', frames)
            module_name = 'module_name' in cfg_parser[section]
            if module_name:
                module_name = cfg_parser[section]['module_name']
                print(module_name)
            else:
                print('missing required module name for conv module')
            layer_name = 'layer_name' in cfg_parser[section]
            if layer_name:
                layer_name = cfg_parser[section]['layer_name']
                print(layer_name)
            else:
                layer_name = str(len(all_layers) - 1)
            tstride = 'tstride' in cfg_parser[section]
            if tstride:
                tstride = int(cfg_parser[section]['tstride'])
            else:
                tstride = 1
            merge_in = 'merge_in' in cfg_parser[section]
            if merge_in:
                merge_in = int(cfg_parser[section]['merge_in'])
            else:
                merge_in = 0
            share = 'share' in cfg_parser[section]
            no_output = 'no_output' in cfg_parser[section]
            image_input = 'image' in cfg_parser[section]
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            tsize = 'tsize' in cfg_parser[section]
            if tsize:
                tsize = int(cfg_parser[section]['tsize'])
            else:
                tsize = 1
            prev_layer_shape = K.int_shape(all_layers[-1])
            input_channels = prev_layer_shape[-1]
            image_size = prev_layer_shape[-3], prev_layer_shape[-2]
            #            print('conv input image size: ', image_size)

            num_convs = int(frames / tstride)
            if num_convs > 1:
                print('num_convs: ', num_convs)
            inputs_needed = (tstride * (num_convs - 1)) + tsize
            #            inputs_needed = frames + tsize - 1
            if inputs_needed > 1:
                print('inputs_needed: ', inputs_needed)
            old_frames_to_read = inputs_needed - frames
            if old_frames_to_read < 0:
                print('negative number of old frames!!!!!!!!!')
            if old_frames_to_read:
                print('history frames needed: ', old_frames_to_read)
            new_frames_to_save = min(frames, old_frames_to_read)
            if new_frames_to_save:
                print('new frames to save: ', new_frames_to_save)

            # attach output ports to inputs we will need next pass
            if no_output is False:
                for f in range(new_frames_to_save):
                    out_index.append(len(all_layers) - frames + f)
                    out_names.append(module_name + '_save_' + str(f))

            # attach output ports to unsaved inputs if we need to share inputs to a slave network
            if share is True:
                for f in range(new_frames_to_save, frames):
                    out_index.append(len(all_layers) - frames + f)
                    out_names.append(module_name + '_share_' + str(f))

            # create input ports for required old frames
            for f in range(old_frames_to_read):
                xx = module_name + '_history_' + str(f)
                in_names.append(xx)
                if image_input:
                    image_inputs.append(xx)
                all_layers.append(Input(shape=(image_size[0], image_size[1], input_channels),
                                        name=xx))
                #                print('History input at: ', len(all_layers) - 1)
                in_index.append(len(all_layers) - 1)

            # create input ports for merged-in frames
            if merge_in > 0:
                input_channels = input_channels + merge_in
                for f in range(inputs_needed):
                    xx = module_name + '_merge_in_' + str(f)
                    in_names.append(xx)
                    all_layers.append(Input(shape=(image_size[0], image_size[1], merge_in),
                                            name=xx))
                    print('merge_in input at: ', len(all_layers) - 1)
                    in_index.append(len(all_layers) - 1)

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # extract parameter for this module from Pytorch checkpoint file
            conv_weights_pt = np.random.rand(input_channels, filters, tsize, size, size)
            conv_bias = [0]
            if module_name + '.weight' in weights_full:
                conv_weights_pt = weights_full[module_name + '.weight']
                print("weight: ", module_name + '.weight', weights_full[module_name + '.weight'].shape)
                # convert to tsize list of 2d conv weight matrices, transposed for Keras
                w_list = []
                if len(conv_weights_pt.shape) == 5:  # check if this is a 3D conv being unfolded
                    for t in range(tsize):
                        w_list.append(
                            np.transpose(conv_weights_pt[:, :, tsize - 1 - t, :, :], [2, 3, 1, 0]))
                else:  # this is simply a single 2D conv
                    w_list.append(np.transpose(conv_weights_pt[:, :, :, :], [2, 3, 1, 0]))
                # concatenate along the in_dim axis the tsize matrices
                conv_weights = np.concatenate(w_list, axis=2)
                if not batch_normalize:
                    conv_bias = weights_full[module_name + '.bias']
            else:
                print('cannot find weight: ', module_name + '.weight')
                fake_weights = True
                conv_weights = np.random.rand(size, size, tsize * input_channels, filters)
                conv_bias = np.zeros(filters)

            if batch_normalize:
                bn_bias = weights_full[module_name + '.batchnorm.bias']
                bn_weight = weights_full[module_name + '.batchnorm.weight']
                bn_running_var = weights_full[module_name + '.batchnorm.running_var']
                bn_running_mean = weights_full[module_name + '.batchnorm.running_mean']

                bn_weight_list = [
                    bn_weight,  # scale gamma
                    bn_bias,  # shift beta
                    bn_running_mean,  # running mean
                    bn_running_var  # running var
                ]

            expected_weights_shape = (size, size, tsize * input_channels, filters)
            print('weight shape, expected : ', expected_weights_shape,
                  'checkpoint: ', conv_weights_pt.shape,
                  'created: ', conv_weights.shape)

            if (conv_weights.shape != expected_weights_shape):
                print('weight matrix shape is wrong, making a fake one')
                fake_weights = True
                conv_weights = np.random.rand(size, size, tsize * input_channels, filters)
                conv_bias = np.zeros(filters)

            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            inputs = []
            outputs = []

            for f in range(inputs_needed):
                #                print('input index: ', len(all_layers) - inputs_needed + f, ' shape: ', all_layers[len(all_layers) - inputs_needed + f].shape)
                inputs.append(all_layers[len(all_layers) - inputs_needed + f])
                if merge_in > 0:
                    #                    print('merge input index: ', len(all_layers) - inputs_needed + f, ' shape: ', all_layers[len(all_layers) - (2*inputs_needed) + f].shape)
                    inputs.append(all_layers[len(all_layers) - (2 * inputs_needed) + f])

            # Create Conv3d from Conv2D layers
            if stride > 1:
                for f in range(len(inputs)):
                    inputs[f] = ZeroPadding2D(((1, 0), (1, 0)))(inputs[f])

            for f in range(int(frames / tstride)):
                layers = []
                if tsize > 1:
                    #                    print('concatenate layers:')
                    for t in range(tsize):
                        # offset is constant with f, except if tstride,
                        # then steps by extra step every time through
                        if merge_in == 0:
                            layers.append(inputs[t + (f * (tstride))])
                        else:
                            layers.append(inputs[2 * (t + (f * (tstride)))])
                            layers.append(inputs[2 * (t + (f * (tstride))) + 1])
                    cat_layer = Concatenate()(layers)
                else:
                    if merge_in == 0:
                        cat_layer = inputs[f * (tstride)]
                    else:
                        layers.append(inputs[2 * f * (tstride)])
                        layers.append(inputs[2 * f * (tstride) + 1])
                        cat_layer = Concatenate()(layers)
                #                print(K.int_shape(cat_layer))
                outputs.append((Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=None,
                    padding=padding))(cat_layer))

            if batch_normalize:
                for f in range(int(frames / tstride)):
                    #                    print (all_layers[0-int(frames/tstride)])
                    #                    print(bn_weight_list)
                    outputs[f] = BatchNormalization(weights=bn_weight_list)(outputs[f])

            if activation == 'relu6':
                for f in range(int(frames / tstride)):
                    outputs[f] = ReLU(max_value=6)(outputs[f])
            elif activation == 'leaky':
                for f in range(int(frames / tstride)):
                    if not conversion_parameters['use_prelu']:
                        outputs[f] = LeakyReLU(alpha=0.1)(outputs[f])
                    else:
                        outputs[f] = PReLU(
                            alpha_initializer=RandomNormal(mean=0.1, stddev=0.0, seed=None),
                            shared_axes=[1, 2])(outputs[f])

            for f in range(int(frames / tstride)):
                all_layers.append(outputs[f])

            frames = int(frames / tstride)
            if frames == 0:
                raise ValueError('tried to time stride single frame')

            layer_names[layer_name] = len(all_layers) - 1

        elif section.startswith('InvResidual'):

            module_name = 'module_name' in cfg_parser[section]
            if module_name:
                module_name = cfg_parser[section]['module_name']
                print(module_name)
            else:
                print('missing required module name for conv module')
            layer_name = 'layer_name' in cfg_parser[section]
            if layer_name:
                layer_name = cfg_parser[section]['layer_name']
                print(layer_name)
            else:
                layer_name = str(len(all_layers) - 1)
            out_channels = int(cfg_parser[section]['out_channels'])
            xratio = int(cfg_parser[section]['xratio'])
            size = int(cfg_parser[section]['size'])
            shift = 'shift' in cfg_parser[section]
            stride = int(cfg_parser[section]['stride'])
            tstride = int(cfg_parser[section]['tstride'])
            print('frames: ', frames)

            frames, fake_weights = invResidual(module_name, layer_name, frames,
                                               out_channels, xratio, size, stride, shift, tstride,
                                               fake_weights)


        elif section.startswith('Linear'):
            module_name = 'module_name' in cfg_parser[section]
            if module_name:
                module_name = cfg_parser[section]['module_name']
                print(module_name)
            else:
                print('missing required module name for Linear module')
            layer_name = 'layer_name' in cfg_parser[section]
            if layer_name:
                layer_name = cfg_parser[section]['layer_name']
                print(layer_name)
            else:
                layer_name = str(len(all_layers) - 1)
            share = 'share' in cfg_parser[section]
            merge_in = 'merge_in' in cfg_parser[section]
            if merge_in:
                merge_in = int(cfg_parser[section]['merge_in'])
            else:
                merge_in = 0
            no_output = 'no_output' in cfg_parser[section]
            outputs = int(cfg_parser[section]['outputs'])
            prev_layer_shape = K.int_shape(all_layers[-1])
            print('prev_layer_shape: ', prev_layer_shape)
            # if share, create output port with all its inputs
            if share is True:
                out_index.append(len(all_layers) - frames)
                out_names.append(module_name + '_share')
            # create input ports for merged-in data
            if merge_in > 0:
                input_channels = input_channels + merge_in
                in_names.append(module_name + '_merge_in')
                all_layers.append(Input(shape=[merge_in], name=module_name + '_merge_in'))
                print('merge_in input at: ', len(all_layers) - 1, ' shape: ', all_layers[-1].shape,
                      ' plus: ', all_layers[-2].shape)
                in_index.append(len(all_layers) - 1)
                layers = []
                layers.append(all_layers[-1])
                layers.append(all_layers[-2])
                all_layers.append(Concatenate()(layers))

            size = np.prod(all_layers[-1].shape[1])  # skip the junk first dimension
            if module_name + '.weight' in weights_full:
                weights = np.transpose(weights_full[module_name + '.weight'], (1, 0))
                bias = weights_full[module_name + '.bias']
            else:
                print('weights missing')
                print('Using fake weights for Linear layer')
                weights = np.random.rand(size, outputs)
                bias = np.random.rand(outputs)
                fake_weights = True
            print('total input size: ', size, 'output size: ', outputs, 'weights: ', weights.shape)
            if (weights.shape[0], weights.shape[1]) != (size, outputs):
                fake_weights = True
                print('Using fake weights for Linear layer')
                weights = np.random.rand(size, outputs)
            if bias.shape != (outputs,):
                fake_weights = True
                print('Using fake bias for Linear layer')
                bias = np.random.rand(outputs)
            weights = [weights, bias]
            print(all_layers[-1])
            all_layers.append(Dense(outputs, weights=weights)(all_layers[-1]))
            layer_names[layer_name] = len(all_layers) - 1

        elif section.startswith('NBLinear'):
            module_name = 'module_name' in cfg_parser[section]
            if module_name:
                module_name = cfg_parser[section]['module_name']
                print(module_name)
            else:
                print('missing required module name for Linear module')
            layer_name = 'layer_name' in cfg_parser[section]
            if layer_name:
                layer_name = cfg_parser[section]['layer_name']
                print(layer_name)
            else:
                layer_name = str(len(all_layers) - 1)
            share = 'share' in cfg_parser[section]
            merge_in = 'merge_in' in cfg_parser[section]
            if merge_in:
                merge_in = int(cfg_parser[section]['merge_in'])
            else:
                merge_in = 0
            no_output = 'no_output' in cfg_parser[section]
            outputs = int(cfg_parser[section]['outputs'])
            prev_layer_shape = K.int_shape(all_layers[-1])
            print('prev_layer_shape: ', prev_layer_shape)
            # if share, create output port with all its inputs
            if share is True:
                out_index.append(len(all_layers) - frames)
                out_names.append(module_name + '_share')
            # create input ports for merged-in data
            if merge_in > 0:
                input_channels = input_channels + merge_in
                in_names.append(module_name + '_merge_in')
                all_layers.append(Input(shape=[merge_in], name=module_name + '_merge_in'))
                print('merge_in input at: ', len(all_layers) - 1, ' shape: ', all_layers[-1].shape,
                      ' plus: ', all_layers[-2].shape)
                in_index.append(len(all_layers) - 1)
                layers = []
                layers.append(all_layers[-1])
                layers.append(all_layers[-2])
                all_layers.append(Concatenate()(layers))

            size = np.prod(all_layers[-1].shape[1])  # skip the junk first dimension
            if module_name + '.weight' in weights_full:
                weights = np.transpose(weights_full[module_name + '.weight'], (1, 0))
            else:
                print('weights missing')
                print('Using fake weights for Linear layer')
                weights = np.random.rand(size, outputs)
                fake_weights = True
            print('total input size: ', size, 'output size: ', outputs, 'weights: ', weights.shape)
            if (weights.shape[0], weights.shape[1]) != (size, outputs):
                fake_weights = True
                print('Using fake weights for Linear layer')
                weights = np.random.rand(size, outputs)
            bias = np.zeros(outputs)
            weights = [weights, bias]
            print(all_layers[-1])
            all_layers.append(Dense(outputs, weights=weights)(all_layers[-1]))
            layer_names[layer_name] = len(all_layers) - 1

        # section 'lookup' just to test finding names
        elif section.startswith('lookup'):
            ids = []
            if 'names' in cfg_parser[section]:
                ids = [layer_names[s.strip()] for s in cfg_parser[section]['names'].split(',')]
            if 'layers' in cfg_parser[section]:
                for i in cfg_parser[section]['layers'].split(','):
                    if int(i) < 0:
                        i = len(all_layers) + int(i)
                    ids.append(int(i))
            print('lookup: ', ids)

        elif section.startswith('route'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            ids = []
            if 'names' in cfg_parser[section]:
                ids = [layer_names[s.strip()] for s in cfg_parser[section]['names'].split(',')]
                print('route from: ', ids)
            if 'layers' in cfg_parser[section]:
                for i in cfg_parser[section]['layers'].split(','):
                    ids.append(int(i))
            layers = [all_layers[i] for i in ids]
            for l in layers:
                print(K.int_shape(l))
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
            else:
                print(layers[0])
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
            layer_names[layer_name] = len(all_layers) - 1
            prev_layer_shape = K.int_shape(all_layers[-1])
            image_size = prev_layer_shape[-2]
            print('route image size: ', image_size)

        elif section.startswith('maxpool'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            for f in range(frames):
                all_layers.append(
                    MaxPooling2D(
                        pool_size=(size, size),
                        strides=(stride, stride),
                        padding='same')(all_layers[0 - frames]))
            layer_names[layer_name] = len(all_layers) - 1
            prev_layer_shape = K.int_shape(all_layers[-1])
            image_size = prev_layer_shape[-2]
            print('maxpool image size: ', image_size)

        elif section.startswith('shortcut'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            if 'name' in cfg_parser[section]:
                index = layer_names[cfg_parser[section]['name'].strip()] - len(all_layers) - 1
            if 'from' in cfg_parser[section]:
                index = frames * int(cfg_parser[section]['from'])
                if (index < 0):
                    print('shortcut index: ', index)
                else:
                    print(
                        'warning: positive absolute layer reference number, I assume you know what you want')
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            for f in range(frames):
                all_layers.append(Add()([all_layers[index], all_layers[0 - frames]]))
            layer_names[layer_name] = len(all_layers) - 1
            prev_layer_shape = K.int_shape(all_layers[-1])
            image_size = prev_layer_shape[-2]
            print('shortcut image size: ', image_size)

        elif section.startswith('upsample'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            for f in range(frames):
                all_layers.append(UpSampling2D(stride)(all_layers[0 - frames]))
            layer_names[layer_name] = len(all_layers) - 1
            prev_layer_shape = K.int_shape(all_layers[-1])
            image_size = prev_layer_shape[-2]
            print('upsample image size: ', image_size)

        elif section.startswith('globalaveragepool'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            for f in range(frames):
                all_layers.append(GlobalAveragePooling2D()(all_layers[0 - frames]))
            layer_names[layer_name] = len(all_layers) - 1
            prev_layer_shape = K.int_shape(all_layers[-1])
            image_size = prev_layer_shape[-2]
            print('global average pooling: ', image_size)

        elif section.startswith('Softmax'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            for f in range(frames):
                all_layers.append(Softmax()(all_layers[0 - frames]))
            layer_names[layer_name] = len(all_layers) - 1
            prev_layer_shape = K.int_shape(all_layers[-1])
            image_size = prev_layer_shape[-2]
            print('softmax: ', image_size)


        elif section.startswith('yolo'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            out_index.append(len(all_layers) - 1)
            out_names.append('yolo_out_' + layer_name)
            all_layers.append(None)
            layer_names[layer_name] = len(all_layers) - 1

        elif section.startswith('output'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
                out_index.append(len(all_layers) - 1)
                out_names.append('output_' + layer_name)
                # all_layers.append(None)
                layer_names[layer_name] = len(all_layers) - 1
            else:
                layer_name = coreml_list[-1][1][0] + '_output'
                out_index.append(len(all_layers) - 1)
                out_names.append(layer_name)
                all_layers.append(None)
                layer_names[layer_name] = len(all_layers) - 1

        elif section.startswith('Qoutput'):
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
                out_index.append(len(all_layers) - 4)
                out_names.append('output4_' + layer_name)
                out_index.append(len(all_layers) - 3)
                out_names.append('output3_' + layer_name)
                out_index.append(len(all_layers) - 2)
                out_names.append('output2_' + layer_name)
                out_index.append(len(all_layers) - 1)
                out_names.append('output1_' + layer_name)
                layer_names[layer_name] = len(all_layers) - 1
            else:
                layer_name = coreml_list[-1][1][0] + '_output'
                out_index.append(len(all_layers) - 1)
                out_names.append(layer_name)
                all_layers.append(None)
                layer_names[layer_name] = len(all_layers) - 1

        elif section.startswith('input'):
            frames = frames + 1
            if 'size' in cfg_parser[section]:
                size = []
                # size = [s.strip() for s in cfg_parser[section]['size'].split(',')]
                for i in cfg_parser[section]['size'].split(','):
                    if i == 'None':
                        size.append(None)
                    else:
                        size.append(int(i))
                print('size: ', size)
            if 'layer_name' in cfg_parser[section]:
                layer_name = cfg_parser[section]['layer_name']
            else:
                layer_name = str(len(all_layers) - 1)
            input_layer = Input(shape=size, name=layer_name)
            in_names.append(layer_name)
            all_layers.append(input_layer)
            if 'image' in cfg_parser[section]:
                image_inputs.append(layer_name)
            print('input layer: ', layer_name, ' shape: ', input_layer.shape)
            in_index.append(len(all_layers) - 1)
            coreml_list.append(('fake', (layer_name), {}, layer_name + ':0'))

        elif section.startswith('net'):
            pass

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    print('done reading config file')
    # assume the end of the network is an output if none are define.
    if len(out_index) == 0:
        print(
            'No outputs defined, so we are assuming last layer is the output and define it as such')
        out_index.append(len(all_layers) - 1)
    model = Model(inputs=[all_layers[i] for i in in_index],
                  outputs=[all_layers[i] for i in out_index])
    print('done assembling model')
    print(model.summary())

    # print all inputs, formatted for use with coremltools Keras convertor
    print('input_names=[')
    for name in in_names:
        print("'" + name + "',")
    print('],')

    # print all outputs, formatted for use with coremltools Keras convertor
    print('output_names=[')
    for name in out_names:
        print("'" + name + "',")
    print('],')

    # Just for fun, print all inputs and outputs and their shapes.

    # Inputs are actual Keras layers so we extract their names
    # also build input_features for CoreML generation
    for layer in [all_layers[i] for i in in_index]:
        print('name: ', layer.name, '; shape: ', layer.shape)

    # Outputs are not Keras layers, so we have a separate list of names for them
    # - name comes from our list,
    # - shape comes from the Keras layer they point to
    out_layers = [all_layers[i] for i in out_index]
    for i in range(len(out_names)):
        print('name: ', out_names[i] + '; shape: ', out_layers[i].shape)

    model.save('{}'.format(keras_file))
    print('Saved Keras model to {}'.format(keras_file))

    # Check to see if all weights have been read.
    #    remaining_weights = len(weights_file.read()) / 4

    #    if remaining_weights > 0:
    #        print('Warning: {} unused weights'.format(remaining_weights))

    if fake_weights == True:
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')
        print('Weights in checkpoint did not match weights required by network')
        print('Fake weights were generated where they were needed!!!!!!!!!!!!')
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')

    if plot_model:
        plot(model, to_file=plot_file, show_shapes=True)
        print('Saved model plot to {}'.format(plot_file))

    build_args = dict()

    print('input_names', in_names)
    print('output_names', out_names)
    print('image_input_names', image_inputs)

    build_args['input_names'] = in_names
    build_args['output_names'] = out_names
    build_args['image_input_names'] = image_inputs

    build_args['use_float_arraytype'] = True

    if float32:
        build_args['model_precision'] = 'float16'

    if conversion_parameters['normalize_inputs']:
        if conversion_parameters['red_bias']:
            build_args['red_bias'] = -conversion_parameters['red_bias'] * 255.0
        if conversion_parameters['green_bias']:
            build_args['green_bias'] = -conversion_parameters['green_bias'] * 255.0
        if conversion_parameters['blue_bias']:
            build_args['blue_bias'] = -conversion_parameters['blue_bias'] * 255.0
        if conversion_parameters['image_scale']:
            if conversion_parameters['image_scale'] != 1.0:
                print('setting image scale this way is not compatible with normalization')
                print('it is happening before bias, which is wrong')
                exit()
    elif conversion_parameters['image_scale']:
        build_args['image_scale'] = 1.0 / conversion_parameters['image_scale']

    coreml_model = coremltools.converters.keras.convert(keras_file, **build_args)
    coreml_model.short_description = coreml_file
    spec = coreml_model.get_spec()

    if conversion_parameters['normalize_inputs']:
        print('\nimage input normalization requested!')
        # get NN portion of the spec
        nn_spec = spec.neuralNetwork
        layers = nn_spec.layers  # this is a list of all the layers
        layers_copy = copy.deepcopy(
            layers)  # make a copy of the layers, these will be added back later
        del nn_spec.layers[:]  # delete all the layers

        for ii in image_inputs:
            # add a scale layer now
            # since mlmodel is in protobuf format, we can add proto messages directly
            # To look at more examples on how to add other layers: see "builder.py" file in coremltools repo

            scale_layer = nn_spec.layers.add()
            scale_layer.name = 'scale_' + ii
            scale_layer.input.append(ii)
            scale_layer.output.append(ii + '_scaled')
            print('inserted scaling layer ', scale_layer.name, )
            print(' input is image input ', ii, ', output is ', ii + '_scaled')

            params = scale_layer.scale
            params.scale.floatValue.extend(
                [conversion_parameters['red_scale'], conversion_parameters['green_scale'], conversion_parameters['blue_scale']])  # scale values for RGB
            params.shapeScale.extend([3, 1, 1])  # shape of the scale vector

        # now add back the rest of the layers (which happens to be just one in this case: the crop layer)
        nn_spec.layers.extend(layers_copy)

        for i in range(len(image_inputs)):
            nn_spec.layers[i + len(image_inputs)].input[0] = image_inputs[i] + '_scaled'
            print('attached layer ', nn_spec.layers[i + len(image_inputs)].name, ' to ',
                  image_inputs[i] + '_scaled')

        # print(spec.description)

        coreml_model = coremltools.models.MLModel(spec)
        coreml_model.short_description = coreml_file
        print('\nsaving normalized network ', coreml_file)
        coreml_model.save(coreml_file)

    else:
        print('\nsaving ', coreml_file)
        coreml_model.save(coreml_file)

    if fake_weights == True:
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')
        print('Weights in checkpoint did not match weights required by network')
        print('Fake weights were generated where they were needed!!!!!!!!!!!!')
        print('************************* Warning!! **************************')
        print('************************* Warning!! **************************')


def finalize_custom_classifier_config(classifier_settings, path_in, backbone_name):
    # if custom classifier, fill the classifier settings with arguments
    if not path_in:
        raise Exception('You have to provide the directory used to train a custom classifier')

    weights_file = os.path.join(path_in, "classifier.checkpoint")
    if not os.path.isfile(weights_file):
        raise Exception(f'The trained classifier "classifier.checkpoint" was not found in {path_in}')

    lab2int_file = os.path.join(path_in, "label2int.json")
    if not os.path.isfile(lab2int_file):
        raise Exception(f'The "label2int.json" was not found in {path_in}')
    try:
        num_classes = np.max(list(json.load(open(lab2int_file)).values())) + 1
    except:
        raise Exception(f'Error parsing "label2int.json"')

    classifier_settings['corresponding_backbone'] = backbone_name
    classifier_settings['weights_file'] = weights_file
    classifier_settings["placeholder_values"]['NUM_CLASSES'] = str(num_classes)

    return classifier_settings

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    backbone_name = args['--backbone']
    classifier_name = args['--classifier']
    output_name = args['--output_name']
    float32 = args['--float32']
    path_in = args['--path_in']
    plot_model = args['--plot_model']

    backbone_settings = SUPPORTED_BACKBONE_CONVERSIONS.get(backbone_name)
    if not backbone_settings:
        raise Exception('Backbone not found: {}. Only the following backbones '
                        'can be converted: {}'.format(backbone_name,
                                                      SUPPORTED_BACKBONE_CONVERSIONS.keys()))

    classifier_settings = SUPPORTED_CLASSIFIER_CONVERSIONS.get(classifier_name)
    if not classifier_settings:
        raise Exception('Classifier not found: {}. Only the following backbones '
                        'can be converted: {}'.format(classifier_name,
                                                      SUPPORTED_CLASSIFIER_CONVERSIONS.keys()))
    if classifier_name == "custom_classifier":
        classifier_settings = finalize_custom_classifier_config(classifier_settings, path_in, backbone_name)

    if classifier_settings['corresponding_backbone'] != backbone_name:
        raise Exception('This classifier expects a different backbone: '
                        '{}'.format(classifier_settings['corresponding_backbone']))

    convert(backbone_settings, classifier_settings, output_name, float32, plot_model)
