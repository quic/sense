
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
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Softmax
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.models import Model

from tools.conversion.section_conversion import invResidual
from tools.conversion.section_conversion import convolutional
from tools.conversion.section_conversion import linear


class ConfigSection:

    def __init__(self, config, all_layers):
        self.module_name = config.get("module_name", None)
        self.layer_name = config["layer_name"] if "layer_name" in config else str(len(all_layers)
                                                                                  - 1)
        self.merge_in = int(config["merge_in"]) if "merge_in" in config else 0
        self.out_channels = int(config["out_channels"]) if "out_channels" in config else None
        self.xratio = int(config["xratio"]) if "xratio" in config else None
        self.size = int(config["size"]) if "size" in config else None
        self.shift = "shift" in config
        self.stride = int(config["stride"]) if "stride" in config else None
        self.tstride = int(config["tstride"]) if "tstride" in config else None
        self.activation = config.get("activation", "")
        self.batch_normalize = "batch_normalize" in config
        self.pad = int(config.get("pad", 1))
        self.tsize = int(config.get("tsize", 1))
        self.filters = int(config["filters"]) if "filters" in config else None
        self.share = "share" in config
        self.no_output = "no_output" in config
        self.image_input = "image" in config
        self.outputs = int(config["outputs"])  if "outputs" in config else None


class Container:

    def __init__(self, conversion_parameters, weights):
        self.all_layers = []
        self.out_index = []
        self.out_names = []
        self.in_index = []
        self.in_names = []
        self.image_inputs = []
        self.layer_names = dict()
        self.frames = 0
        self.coreml_list = []
        self.fake_weights = False
        self.conversion_parameters = conversion_parameters
        self.weights = weights


class KerasConver:

    def __init__(self, cfg_parser, weights_full, conversion_parameters):
        self.cfg_parser = cfg_parser
        self.container = Container(conversion_parameters, weights_full)

    def create_keras_model(self):
        print("Creating Keras model.")
        np.random.seed(13)  # start the same way each time...

        for section in self.cfg_parser.sections():
            config = ConfigSection(self.cfg_parser[section])
            print("    ***** Parsing section {} ************".format(section))
            if section.startswith("convolutional"):
                convolutional(config, self.container)

            elif section.startswith("InvResidual"):
                invResidual(config, self.container)

            elif section.startswith("Linear"):
                linear(config, self.container)

            elif section.startswith("NBLinear"):
                module_name = "module_name" in cfg_parser[section]
                if module_name:
                    module_name = cfg_parser[section]["module_name"]
                    print(module_name)
                else:
                    print("missing required module name for Linear module")
                layer_name = "layer_name" in cfg_parser[section]
                if layer_name:
                    layer_name = cfg_parser[section]["layer_name"]
                    print(layer_name)
                else:
                    layer_name = str(len(all_layers) - 1)
                share = "share" in cfg_parser[section]
                merge_in = "merge_in" in cfg_parser[section]
                if merge_in:
                    merge_in = int(cfg_parser[section]["merge_in"])
                else:
                    merge_in = 0
                no_output = "no_output" in cfg_parser[section]
                outputs = int(cfg_parser[section]["outputs"])
                prev_layer_shape = K.int_shape(all_layers[-1])
                print("prev_layer_shape: ", prev_layer_shape)
                # if share, create output port with all its inputs
                if share is True:
                    out_index.append(len(all_layers) - frames)
                    out_names.append(module_name + "_share")
                # create input ports for merged-in data
                if merge_in > 0:
                    input_channels = input_channels + merge_in
                    in_names.append(module_name + "_merge_in")
                    all_layers.append(
                        Input(shape=[merge_in], name=module_name + "_merge_in")
                    )
                    print(
                        "merge_in input at: ",
                        len(all_layers) - 1,
                        " shape: ",
                        all_layers[-1].shape,
                        " plus: ",
                        all_layers[-2].shape,
                    )
                    in_index.append(len(all_layers) - 1)
                    layers = []
                    layers.append(all_layers[-1])
                    layers.append(all_layers[-2])
                    all_layers.append(Concatenate()(layers))

                size = np.prod(all_layers[-1].shape[1])  # skip the junk first dimension
                if module_name + ".weight" in weights_full:
                    weights = np.transpose(weights_full[module_name + ".weight"], (1, 0))
                else:
                    print("weights missing")
                    print("Using fake weights for Linear layer")
                    weights = np.random.rand(size, outputs)
                    fake_weights = True
                print(
                    "total input size: ",
                    size,
                    "output size: ",
                    outputs,
                    "weights: ",
                    weights.shape,
                )
                if (weights.shape[0], weights.shape[1]) != (size, outputs):
                    fake_weights = True
                    print("Using fake weights for Linear layer")
                    weights = np.random.rand(size, outputs)
                bias = np.zeros(outputs)
                weights = [weights, bias]
                print(all_layers[-1])
                all_layers.append(Dense(outputs, weights=weights)(all_layers[-1]))
                layer_names[layer_name] = len(all_layers) - 1

            # section 'lookup' just to test finding names
            elif section.startswith("lookup"):
                ids = []
                if "names" in cfg_parser[section]:
                    ids = [
                        layer_names[s.strip()]
                        for s in cfg_parser[section]["names"].split(",")
                    ]
                if "layers" in cfg_parser[section]:
                    for i in cfg_parser[section]["layers"].split(","):
                        if int(i) < 0:
                            i = len(all_layers) + int(i)
                        ids.append(int(i))
                print("lookup: ", ids)

            elif section.startswith("route"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                ids = []
                if "names" in cfg_parser[section]:
                    ids = [
                        layer_names[s.strip()]
                        for s in cfg_parser[section]["names"].split(",")
                    ]
                    print("route from: ", ids)
                if "layers" in cfg_parser[section]:
                    for i in cfg_parser[section]["layers"].split(","):
                        ids.append(int(i))
                layers = [all_layers[i] for i in ids]
                for l in layers:
                    print(K.int_shape(l))
                if len(layers) > 1:
                    print("Concatenating route layers:", layers)
                    concatenate_layer = Concatenate()(layers)
                    all_layers.append(concatenate_layer)
                else:
                    print(layers[0])
                    skip_layer = layers[0]  # only one layer to route
                    all_layers.append(skip_layer)
                layer_names[layer_name] = len(all_layers) - 1
                prev_layer_shape = K.int_shape(all_layers[-1])
                image_size = prev_layer_shape[-2]
                print("route image size: ", image_size)

            elif section.startswith("maxpool"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                size = int(cfg_parser[section]["size"])
                stride = int(cfg_parser[section]["stride"])
                for f in range(frames):
                    all_layers.append(
                        MaxPooling2D(
                            pool_size=(size, size), strides=(stride, stride), padding="same"
                        )(all_layers[0 - frames])
                    )
                layer_names[layer_name] = len(all_layers) - 1
                prev_layer_shape = K.int_shape(all_layers[-1])
                image_size = prev_layer_shape[-2]
                print("maxpool image size: ", image_size)

            elif section.startswith("shortcut"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                if "name" in cfg_parser[section]:
                    layer = cfg_parser[section]["name"].strip()
                    index = layer_names[layer] - len(all_layers) - 1
                if "from" in cfg_parser[section]:
                    index = frames * int(cfg_parser[section]["from"])
                    if index < 0:
                        print("shortcut index: ", index)
                    else:
                        print(
                            "warning: positive absolute layer reference number, I assume you know what you want"
                        )
                activation = cfg_parser[section]["activation"]
                assert activation == "linear", "Only linear activation supported."
                for f in range(frames):
                    all_layers.append(Add()([all_layers[index], all_layers[0 - frames]]))
                layer_names[layer_name] = len(all_layers) - 1
                prev_layer_shape = K.int_shape(all_layers[-1])
                image_size = prev_layer_shape[-2]
                print("shortcut image size: ", image_size)

            elif section.startswith("upsample"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                stride = int(cfg_parser[section]["stride"])
                assert stride == 2, "Only stride=2 supported."
                for f in range(frames):
                    all_layers.append(UpSampling2D(stride)(all_layers[0 - frames]))
                layer_names[layer_name] = len(all_layers) - 1
                prev_layer_shape = K.int_shape(all_layers[-1])
                image_size = prev_layer_shape[-2]
                print("upsample image size: ", image_size)

            elif section.startswith("globalaveragepool"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                for f in range(frames):
                    all_layers.append(GlobalAveragePooling2D()(all_layers[0 - frames]))
                layer_names[layer_name] = len(all_layers) - 1
                prev_layer_shape = K.int_shape(all_layers[-1])
                image_size = prev_layer_shape[-2]
                print("global average pooling: ", image_size)

            elif section.startswith("Softmax"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                for f in range(frames):
                    all_layers.append(Softmax()(all_layers[0 - frames]))
                layer_names[layer_name] = len(all_layers) - 1
                prev_layer_shape = K.int_shape(all_layers[-1])
                image_size = prev_layer_shape[-2]
                print("softmax: ", image_size)

            elif section.startswith("yolo"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                out_index.append(len(all_layers) - 1)
                out_names.append("yolo_out_" + layer_name)
                all_layers.append(None)
                layer_names[layer_name] = len(all_layers) - 1

            elif section.startswith("output"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                    out_index.append(len(all_layers) - 1)
                    out_names.append("output_" + layer_name)
                    # all_layers.append(None)
                    layer_names[layer_name] = len(all_layers) - 1
                else:
                    layer_name = coreml_list[-1][1][0] + "_output"
                    out_index.append(len(all_layers) - 1)
                    out_names.append(layer_name)
                    all_layers.append(None)
                    layer_names[layer_name] = len(all_layers) - 1

            elif section.startswith("Qoutput"):
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                    out_index.append(len(all_layers) - 4)
                    out_names.append("output4_" + layer_name)
                    out_index.append(len(all_layers) - 3)
                    out_names.append("output3_" + layer_name)
                    out_index.append(len(all_layers) - 2)
                    out_names.append("output2_" + layer_name)
                    out_index.append(len(all_layers) - 1)
                    out_names.append("output1_" + layer_name)
                    layer_names[layer_name] = len(all_layers) - 1
                else:
                    layer_name = coreml_list[-1][1][0] + "_output"
                    out_index.append(len(all_layers) - 1)
                    out_names.append(layer_name)
                    all_layers.append(None)
                    layer_names[layer_name] = len(all_layers) - 1

            elif section.startswith("input"):
                frames = frames + 1
                if "size" in cfg_parser[section]:
                    size = []
                    for i in cfg_parser[section]["size"].split(","):
                        if i == "None":
                            size.append(None)
                        else:
                            size.append(int(i))
                    print("size: ", size)
                if "layer_name" in cfg_parser[section]:
                    layer_name = cfg_parser[section]["layer_name"]
                else:
                    layer_name = str(len(all_layers) - 1)
                input_layer = Input(shape=size, name=layer_name)
                in_names.append(layer_name)
                all_layers.append(input_layer)
                if "image" in cfg_parser[section]:
                    image_inputs.append(layer_name)
                print("input layer: ", layer_name, " shape: ", input_layer.shape)
                in_index.append(len(all_layers) - 1)
                coreml_list.append(("fake", (layer_name), {}, layer_name + ":0"))

            elif section.startswith("net"):
                pass

            else:
                raise ValueError("Unsupported section header type: {}".format(section))

        # Create and save model.
        print("done reading config file")
        # assume the end of the network is an output if none are define.
        if len(out_index) == 0:
            print(
                "No outputs defined, so we are assuming last layer is the output and define it as such"
            )
            out_index.append(len(all_layers) - 1)
        model = Model(
            inputs=[all_layers[i] for i in in_index],
            outputs=[all_layers[i] for i in out_index],
        )
        print("done assembling model")
        print(model.summary())

        # print all inputs, formatted for use with coremltools Keras convertor
        print("input_names=[")
        for name in in_names:
            print("'" + name + "',")
        print("],")

        # print all outputs, formatted for use with coremltools Keras convertor
        print("output_names=[")
        for name in out_names:
            print("'" + name + "',")
        print("],")

        # Just for fun, print all inputs and outputs and their shapes.

        # Inputs are actual Keras layers so we extract their names
        # also build input_features for CoreML generation
        for layer in [all_layers[i] for i in in_index]:
            print("name: ", layer.name, "; shape: ", layer.shape)

        # Outputs are not Keras layers, so we have a separate list of names for them
        # - name comes from our list,
        # - shape comes from the Keras layer they point to
        out_layers = [all_layers[i] for i in out_index]
        for i in range(len(out_names)):
            print("name: ", out_names[i] + "; shape: ", out_layers[i].shape)
        return model, fake_weights, in_names, out_names, image_inputs
