import logging

import numpy as np

from keras.layers.advanced_activations import ReLU
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import DepthwiseConv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU


def invResidual(config, container):
    if not config.module_name:
        raise ValueError('Missing module name in section')

    config.layer_name = (
        config.layer_name if config.layer_name else str(len(container.all_layers) - 1)
    )
    logging.info(f"frames: {container.frames}")
    s = 0
    if config.shift:
        logging.info("3D conv block")
        tsize = 3
    else:
        logging.info("2D conv block")
        tsize = 1
    config.size = int(config.size)

    prev_layer = container.all_layers[-1]
    prev_layer_shape = K.int_shape(prev_layer)
    input_channels = prev_layer_shape[-1]
    x_channels = input_channels * config.xratio
    image_size = prev_layer_shape[-3], prev_layer_shape[-2]
    logging.info(f"input image size: {image_size}")
    num_convs = int(container.frames / config.tstride)
    inputs_needed = (config.tstride * (num_convs - 1)) + tsize
    #            inputs_needed = frames + tsize - 1
    if inputs_needed > 1:
        logging.info(f"inputs_needed: {inputs_needed}")
    old_frames_to_read = inputs_needed - container.frames
    new_frames_to_save = min(container.frames, old_frames_to_read)
    logging.info(
        f"num_convs: {num_convs}, inputs_needed: {inputs_needed}, history frames needed: {old_frames_to_read},"
        f"frames to save: {new_frames_to_save}, tstride: {config.tstride}"
    )
    # create (optional) expansion pointwise convolution layer

    input_indexes = []
    for i in range(num_convs):
        input_indexes.append(
            len(container.all_layers) - container.frames + (i * config.tstride)
        )

    if config.xratio != 1:
        logging.info("---------- Insert channel multiplier pointwise conv -------------")
        # attach output ports to inputs we will need next pass if tsize>1
        for f in range(new_frames_to_save):
            container.out_index.append(len(container.all_layers) - container.frames + f)
            container.out_names.append(config.module_name + "_save_" + str(f))

        # create input ports for required old frames if tsize>1
        for f in range(old_frames_to_read):
            h_name = config.module_name + "_history_" + str(f)
            container.all_layers.append(
                Input(shape=(image_size[0], image_size[1], input_channels), name=h_name)
            )
            container.in_names.append(h_name)
            container.in_index.append(len(container.all_layers) - 1)

        # get weights
        n = config.module_name + ".conv." + str(s) + ".0."
        if n + "weight" in container.weights:
            weights_pt = container.weights[n + "weight"]
            logging.info(f"checkpoint: {weights_pt.shape}")
            weights_k = np.transpose(weights_pt, [2, 3, 1, 0])
            bias = container.weights[n + "bias"]
        else:
            logging.info(f"missing weight {n}weight")
            weights_k = np.random.rand(1, 1, tsize * input_channels, x_channels)
            bias = np.zeros(x_channels)
            container.fake_weights = True

        expected_weights_shape = (1, 1, tsize * input_channels, x_channels)
        logging.info(
            f"weight shape, expected : {expected_weights_shape} transposed: {weights_k.shape}"
        )

        if weights_k.shape != expected_weights_shape:
            logging.info("weight matrix shape is wrong, making a fake one")
            weights_k = np.random.rand(1, 1, tsize * input_channels, x_channels)
            bias = np.zeros(x_channels)
            container.fake_weights = True

        weights = [weights_k, bias]

        inputs = []
        outputs = []

        for f in range(inputs_needed):
            inputs.append(
                container.all_layers[len(container.all_layers) - inputs_needed + f]
            )
            if config.merge_in > 0:
                inputs.append(
                    container.all_layers[
                        len(container.all_layers) - (2 * inputs_needed) + f
                    ]
                )

        for f in range(int(container.frames / config.tstride)):
            layers = []
            if tsize > 1:
                for t in range(tsize):
                    # offset is constant with f, except if tstride,
                    # then steps by extra step every time through
                    layers.append(inputs[(tsize - t - 1) + (f * (config.tstride))])
                cat_layer = Concatenate()(layers)
            else:
                cat_layer = inputs[f * (config.tstride)]

            outputs.append(
                (
                    Conv2D(
                        x_channels,
                        (1, 1),
                        use_bias=not config.batch_normalize,
                        weights=weights,
                        activation=None,
                        padding="same",
                    )
                )(cat_layer)
            )

        logging.info(
            f"parallel convs: {int(container.frames / config.tstride)} : {K.int_shape(cat_layer)}"
        )

        if config.activation == "leaky":
            for f in range(int(container.frames / config.tstride)):
                if not container.conversion_parameters["use_prelu"]:
                    outputs[f] = LeakyReLU(alpha=0.1)(outputs[f])
                else:
                    outputs[f] = PReLU(
                        alpha_initializer=RandomNormal(mean=0.1, stddev=0.0, seed=None),
                        shared_axes=[1, 2],
                    )(outputs[f])
        elif config.activation == "relu6":
            for f in range(int(container.frames / config.tstride)):
                outputs[f] = ReLU(max_value=6)(outputs[f])

        for f in range(int(container.frames / config.tstride)):
            container.all_layers.append(outputs[f])
        s += 1
        container.frames = int(container.frames / config.tstride)

    else:
        logging.info("Skipping channel multiplier pointwise conv, no expansion")

    # create groupwise convolution
    # get weights
    logging.info("---------- Depthwise conv -------------")
    n = config.module_name + ".conv." + str(s) + ".0."
    logging.info(f"module name base: {n}")
    if n + "weight" in container.weights:
        weights_pt = container.weights[n + "weight"]
        logging.info(f"checkpoint: {weights_pt.shape}")
        weights_k = np.transpose(weights_pt, [2, 3, 0, 1])
        bias = container.weights[n + "bias"]
    else:
        logging.info(f"missing weight {n}weight")
        weights_k = np.random.rand(config.size, config.size, x_channels, 1)
        bias = np.zeros(x_channels)
        container.fake_weights = True

    expected_weights_shape = (config.size, config.size, x_channels, 1)
    logging.info(f"weight shape, expected : {expected_weights_shape} transposed: {weights_k.shape}")

    if weights_k.shape != expected_weights_shape:
        logging.error("weight matrix shape is wrong, making a fake one")
        container.fake_weights = True
        weights_k = np.random.rand(config.size, config.size, x_channels, 1)
        bias = np.zeros(x_channels)

    weights = [weights_k, bias]

    inputs = []
    outputs = []

    padding = "same" if config.pad == 1 and config.stride == 1 else "valid"

    for f in range(container.frames):
        inputs.append(
            container.all_layers[len(container.all_layers) - container.frames + f]
        )

    if config.stride > 1:
        for f in range(len(inputs)):
            if config.size == 3:  # originally for all sizes
                inputs[f] = ZeroPadding2D(
                    ((config.size - config.stride, 0), (config.size - config.stride, 0))
                )(inputs[f])
            elif config.size == 5:  # I found this works...
                inputs[f] = ZeroPadding2D(((2, 2), (2, 2)))(inputs[f])
            else:
                logging.info(f"I have no idea what to do for size {config.size}")
                exit()

    logging.info(f"parallel convs: {f} : {K.int_shape(inputs[0])}, padding: {padding}")
    for f in range(container.frames):
        outputs.append(
            (
                DepthwiseConv2D(
                    (config.size, config.size),
                    strides=(config.stride, config.stride),
                    use_bias=not config.batch_normalize,
                    weights=weights,
                    activation=None,
                    padding=padding,
                )
            )(inputs[f])
        )

    if config.activation == "leaky":
        for f in range(int(container.frames)):
            if not container.conversion_parameters["use_prelu"]:
                outputs[f] = LeakyReLU(alpha=0.1)(outputs[f])
            else:
                outputs[f] = PReLU(
                    alpha_initializer=RandomNormal(mean=0.1, stddev=0.0, seed=None),
                    shared_axes=[1, 2],
                )(outputs[f])
    elif config.activation == "relu6":
        for f in range(int(container.frames)):
            outputs[f] = ReLU(max_value=6)(outputs[f])

    for f in range(int(container.frames)):
        container.all_layers.append(outputs[f])
    s += 1

    # create pointwise convolution
    # get weights
    logging.info("---------- Pointwise conv -------------")
    n = config.module_name + ".conv." + str(s) + "."
    logging.info(f"module name base: {n}")
    if n + "weight" in container.weights:
        weights_pt = container.weights[n + "weight"]
        logging.info(f"checkpoint: {weights_pt.shape}")
        weights_k = np.transpose(weights_pt, [2, 3, 1, 0])
        bias = container.weights[n + "bias"]
    else:
        logging.error(f"missing weight {n}weight")
        container.fake_weights = True
        weights_k = np.random.rand(1, 1, x_channels, config.out_channels)
        bias = np.zeros(config.out_channels)

    expected_weights_shape = (1, 1, x_channels, config.out_channels)
    logging.info(
        f"weight shape, expected : {expected_weights_shape}"
        f"transposed: {weights_k.shape}"
    )

    if weights_k.shape != expected_weights_shape:
        logging.error("weight matrix shape is wrong, making a fake one")
        container.fake_weights = True
        weights_k = np.random.rand(1, 1, x_channels, config.out_channels)
        bias = np.zeros(config.out_channels)

    weights = [weights_k, bias]
    logging.info(f"combined shape: {weights[0].shape} {weights[1].shape}")

    inputs = []
    outputs = []

    for f in range(container.frames):
        inputs.append(
            container.all_layers[len(container.all_layers) - container.frames + f]
        )

    shape = K.int_shape(container.all_layers[len(container.all_layers) - container.frames])
    logging.info(f"parallel convs: {f} : {shape}")
    for f in range(container.frames):
        conv_input = container.all_layers[
            len(container.all_layers) - container.frames + f
        ]

        outputs.append(
            (
                Conv2D(
                    config.out_channels,
                    (1, 1),
                    use_bias=not config.batch_normalize,
                    weights=weights,
                    activation=None,
                    padding="same",
                )
            )(conv_input)
        )

    if config.stride == 1 and input_channels == config.out_channels:
        for f in range(int(container.frames)):
            container.all_layers.append(
                Add()([container.all_layers[input_indexes[f]], outputs[f]])
            )
    else:
        for f in range(int(container.frames)):
            container.all_layers.append(outputs[f])
    s += 1


def convolutional(config, container):
    if not config.module_name:
        raise ValueError('Missing module name in section')

    config.layer_name = (
        config.layer_name if config.layer_name else str(len(container.all_layers) - 1)
    )
    config.size = int(config.size)
    if container.frames > 1:
        logging.info(f"frames: {container.frames}")
    prev_layer_shape = K.int_shape(container.all_layers[-1])
    input_channels = prev_layer_shape[-1]
    image_size = prev_layer_shape[-3], prev_layer_shape[-2]

    num_convs = int(container.frames / config.tstride)
    if num_convs > 1:
        logging.info(f"num_convs: {num_convs}")
    inputs_needed = (config.tstride * (num_convs - 1)) + config.tsize
    #            inputs_needed = frames + tsize - 1
    if inputs_needed > 1:
        logging.info(f"inputs_needed: {inputs_needed}")
    old_frames_to_read = inputs_needed - container.frames
    if old_frames_to_read < 0:
        logging.info("negative number of old frames!!!!!!!!!")
    if old_frames_to_read:
        logging.info(f"history frames needed: {old_frames_to_read}")
    new_frames_to_save = min(container.frames, old_frames_to_read)
    if new_frames_to_save:
        logging.info(f"new frames to save: {new_frames_to_save}")

    # attach output ports to inputs we will need next pass
    if config.no_output is False:
        for f in range(new_frames_to_save):
            container.out_index.append(len(container.all_layers) - container.frames + f)
            container.out_names.append(config.module_name + "_save_" + str(f))

    # attach output ports to unsaved inputs if we need to share inputs to a slave network
    if config.share is True:
        for f in range(new_frames_to_save, container.frames):
            container.out_index.append(len(container.all_layers) - container.frames + f)
            container.out_names.append(config.module_name + "_share_" + str(f))

    # create input ports for required old frames
    for f in range(old_frames_to_read):
        xx = config.module_name + "_history_" + str(f)
        container.in_names.append(xx)
        if config.image_input:
            container.image_inputs.append(xx)
        container.all_layers.append(
            Input(shape=(image_size[0], image_size[1], input_channels), name=xx)
        )
        container.in_index.append(len(container.all_layers) - 1)

    # create input ports for merged-in frames
    if config.merge_in > 0:
        input_channels = input_channels + config.merge_in
        for f in range(inputs_needed):
            xx = config.module_name + "_merge_in_" + str(f)
            container.in_names.append(xx)
            container.all_layers.append(
                Input(shape=(image_size[0], image_size[1], config.merge_in), name=xx)
            )
            logging.info(f"merge_in input at: {len(container.all_layers) - 1}")
            container.in_index.append(len(container.all_layers) - 1)

    padding = "same" if config.pad == 1 and config.stride == 1 else "valid"

    # extract parameter for this module from Pytorch checkpoint file
    conv_weights_pt = np.random.rand(
        input_channels, config.filters, config.tsize, config.size, config.size
    )
    conv_bias = [0]
    if config.module_name + ".weight" in container.weights:
        conv_weights_pt = container.weights[config.module_name + ".weight"]
        shape = container.weights[config.module_name + ".weight"].shape
        logging.info(f"weight: {config.module_name}.weight {shape}")
        # convert to tsize list of 2d conv weight matrices, transposed for Keras
        w_list = []
        if len(conv_weights_pt.shape) == 5:  # check if this is a 3D conv being unfolded
            for t in range(config.tsize):
                w_list.append(
                    np.transpose(
                        conv_weights_pt[:, :, config.tsize - 1 - t, :, :], [2, 3, 1, 0]
                    )
                )
        else:  # this is simply a single 2D conv
            w_list.append(np.transpose(conv_weights_pt[:, :, :, :], [2, 3, 1, 0]))
        # concatenate along the in_dim axis the tsize matrices
        conv_weights = np.concatenate(w_list, axis=2)
        if not config.batch_normalize:
            conv_bias = container.weights[config.module_name + ".bias"]
    else:
        logging.info(f"cannot find weight: {config.module_name}.weight")
        container.fake_weights = True
        conv_weights = np.random.rand(
            config.size, config.size, config.tsize * input_channels, config.filters
        )
        conv_bias = np.zeros(config.filters)

    if config.batch_normalize:
        bn_bias = container.weights[config.module_name + ".batchnorm.bias"]
        bn_weight = container.weights[config.module_name + ".batchnorm.weight"]
        bn_running_var = container.weights[
            config.module_name + ".batchnorm.running_var"
        ]
        bn_running_mean = container.weights[
            config.module_name + ".batchnorm.running_mean"
        ]

        bn_weight_list = [
            bn_weight,  # scale gamma
            bn_bias,  # shift beta
            bn_running_mean,  # running mean
            bn_running_var,  # running var
        ]

    expected_weights_shape = (
        config.size,
        config.size,
        config.tsize * input_channels,
        config.filters,
    )
    logging.info(
        f"weight shape, expected : {expected_weights_shape} "
        f"checkpoint: {conv_weights_pt.shape} "
        f"created: {conv_weights.shape} "
    )

    if conv_weights.shape != expected_weights_shape:
        logging.info("weight matrix shape is wrong, making a fake one")
        container.fake_weights = True
        conv_weights = np.random.rand(
            config.size, config.size, config.tsize * input_channels, config.filters
        )
        conv_bias = np.zeros(config.filters)

    conv_weights = (
        [conv_weights] if config.batch_normalize else [conv_weights, conv_bias]
    )

    inputs = []
    outputs = []

    for f in range(inputs_needed):
        inputs.append(
            container.all_layers[len(container.all_layers) - inputs_needed + f]
        )
        if config.merge_in > 0:
            inputs.append(
                container.all_layers[
                    len(container.all_layers) - (2 * inputs_needed) + f
                ]
            )

    # Create Conv3d from Conv2D layers
    if config.stride > 1:
        for f in range(len(inputs)):
            inputs[f] = ZeroPadding2D(((1, 0), (1, 0)))(inputs[f])

    for f in range(int(container.frames / config.tstride)):
        layers = []
        if config.tsize > 1:
            for t in range(config.tsize):
                # offset is constant with f, except if tstride,
                # then steps by extra step every time through
                if config.merge_in == 0:
                    layers.append(inputs[t + (f * (config.tstride))])
                else:
                    layers.append(inputs[2 * (t + (f * (config.tstride)))])
                    layers.append(inputs[2 * (t + (f * (config.tstride))) + 1])
            cat_layer = Concatenate()(layers)
        else:
            if config.merge_in == 0:
                cat_layer = inputs[f * (config.tstride)]
            else:
                layers.append(inputs[2 * f * (config.tstride)])
                layers.append(inputs[2 * f * (config.tstride) + 1])
                cat_layer = Concatenate()(layers)
        outputs.append(
            (
                Conv2D(
                    config.filters,
                    (config.size, config.size),
                    strides=(config.stride, config.stride),
                    kernel_regularizer=l2(5e-4),
                    use_bias=not config.batch_normalize,
                    weights=conv_weights,
                    activation=None,
                    padding=padding,
                )
            )(cat_layer)
        )

    if config.batch_normalize:
        for f in range(int(container.frames / config.tstride)):
            outputs[f] = BatchNormalization(weights=bn_weight_list)(outputs[f])

    if config.activation == "relu6":
        for f in range(int(container.frames / config.tstride)):
            outputs[f] = ReLU(max_value=6)(outputs[f])
    elif config.activation == "leaky":
        for f in range(int(container.frames / config.tstride)):
            if not container.conversion_parameters["use_prelu"]:
                outputs[f] = LeakyReLU(alpha=0.1)(outputs[f])
            else:
                outputs[f] = PReLU(
                    alpha_initializer=RandomNormal(mean=0.1, stddev=0.0, seed=None),
                    shared_axes=[1, 2],
                )(outputs[f])

    for f in range(int(container.frames / config.tstride)):
        container.all_layers.append(outputs[f])

    frames = int(container.frames / config.tstride)
    if frames == 0:
        raise ValueError("tried to time stride single frame")

    container.layer_names[config.layer_name] = len(container.all_layers) - 1


def linear(config, container):
    if not config.module_name:
        raise ValueError('Missing module name in section')

    config.layer_name = (
        config.layer_name if config.layer_name else str(len(container.all_layers) - 1)
    )
    prev_layer_shape = K.int_shape(container.all_layers[-1])
    input_channels = prev_layer_shape[-1]

    logging.info(f"prev_layer_shape: {prev_layer_shape}")
    # if share, create output port with all its inputs
    if config.share is True:
        container.out_index.append(len(container.all_layers) - container.frames)
        container.out_names.append(config.module_name + "_share")
    # create input ports for merged-in data
    if config.merge_in > 0:
        input_channels = input_channels + config.merge_in
        container.in_names.append(config.module_name + "_merge_in")
        container.all_layers.append(
            Input(shape=[config.merge_in], name=config.module_name + "_merge_in")
        )
        logging.info(
            f"merge_in input at: {len(container.all_layers) - 1} "
            f"shape: {container.all_layers[-1].shape} "
            f"plus: {container.all_layers[-2].shape}"
        )
        container.in_index.append(len(container.all_layers) - 1)
        layers = []
        layers.append(container.all_layers[-1])
        layers.append(container.all_layers[-2])
        container.all_layers.append(Concatenate()(layers))

    size = np.prod(container.all_layers[-1].shape[1])  # skip the junk first dimension
    if config.module_name + ".weight" in container.weights:
        weights = np.transpose(
            container.weights[config.module_name + ".weight"], (1, 0)
        )
        bias = container.weights[config.module_name + ".bias"]
    else:
        logging.error("weights missing")
        logging.error("Using fake weights for Linear layer")
        weights = np.random.rand(size, config.outputs)
        bias = np.random.rand(config.outputs)
        container.fake_weights = True
    logging.info(
        f"total input size: {size} "
        f"output size: {config.outputs} "
        f"weights: {weights.shape}"
    )
    if (weights.shape[0], weights.shape[1]) != (size, config.outputs):
        container.fake_weights = True
        logging.error("Using fake weights for Linear layer")
        weights = np.random.rand(size, config.outputs)
    if bias.shape != (config.outputs,):
        container.fake_weights = True
        logging.error("Using fake bias for Linear layer")
        bias = np.random.rand(config.outputs)
    weights = [weights, bias]
    logging.info(container.all_layers[-1])
    container.all_layers.append(
        Dense(config.outputs, weights=weights)(container.all_layers[-1])
    )
    container.layer_names[config.layer_name] = len(container.all_layers) - 1


def globalaveragepool(config, container):
    config.layer_name = (
        config.layer_name if config.layer_name else str(len(container.all_layers) - 1)
    )
    for f in range(container.frames):
        container.all_layers.append(
            GlobalAveragePooling2D()(container.all_layers[0 - container.frames])
        )
    container.layer_names[config.layer_name] = len(container.all_layers) - 1
    prev_layer_shape = K.int_shape(container.all_layers[-1])
    image_size = prev_layer_shape[-2]
    logging.info(f"global average pooling: {image_size}")


def input(config, container):
    config.layer_name = (
        config.layer_name if config.layer_name else str(len(container.all_layers) - 1)
    )
    container.frames = container.frames + 1
    size = []
    for i in config.size.split(","):
        if i == "None":
            size.append(None)
        else:
            size.append(int(i))
    logging.info(f"size: {size}")
    input_layer = Input(shape=size, name=config.layer_name)
    container.in_names.append(config.layer_name)
    container.all_layers.append(input_layer)
    if config.image_input:
        container.image_inputs.append(config.layer_name)
    logging.info(f"input layer: {config.layer_name} shape: {input_layer.shape}")
    container.in_index.append(len(container.all_layers) - 1)
    container.layer_list.append(config.layer_name)


def output(config, container):
    if config.layer_name:
        layer_name = config.layer_name
        container.out_index.append(len(container.all_layers) - 1)
        container.out_names.append("output_" + layer_name)
        # all_layers.append(None)
        container.layer_names[layer_name] = len(container.all_layers) - 1
    else:
        layer_name = container.layer_list[-1] + "_output"
        container.out_index.append(len(container.all_layers) - 1)
        container.out_names.append(layer_name)
        container.all_layers.append(None)
        container.layer_names[layer_name] = len(container.all_layers) - 1
