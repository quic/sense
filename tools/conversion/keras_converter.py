import logging

import numpy as np
from keras.models import Model

from tools.conversion.section_conversion import invResidual
from tools.conversion.section_conversion import convolutional
from tools.conversion.section_conversion import linear
from tools.conversion.section_conversion import globalaveragepool
from tools.conversion.section_conversion import input
from tools.conversion.section_conversion import output


class ConfigSection:
    def __init__(self, config):
        self.module_name = config.get("module_name", None)
        self.layer_name = config.get("layer_name", "")
        self.merge_in = int(config["merge_in"]) if "merge_in" in config else 0
        self.out_channels = (
            int(config["out_channels"]) if "out_channels" in config else None
        )
        self.xratio = int(config["xratio"]) if "xratio" in config else None
        self.size = config["size"] if "size" in config else None
        self.shift = "shift" in config
        self.stride = int(config["stride"]) if "stride" in config else None
        self.tstride = int(config["tstride"]) if "tstride" in config else 1
        self.activation = config.get("activation", "")
        self.batch_normalize = "batch_normalize" in config
        self.pad = int(config.get("pad", 1))
        self.tsize = int(config.get("tsize", 1))
        self.filters = int(config["filters"]) if "filters" in config else None
        self.share = "share" in config
        self.no_output = "no_output" in config
        self.image_input = "image" in config
        self.outputs = int(config["outputs"]) if "outputs" in config else None


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
        self.layer_list = []
        self.fake_weights = False
        self.conversion_parameters = conversion_parameters
        self.weights = weights


class KerasConverter:
    def __init__(self, cfg, weights, conversion_parameters):
        self.cfg_parser = cfg
        self.container = Container(conversion_parameters, weights)

    def create_keras_model(self):
        logging.info("Creating Keras model.")
        np.random.seed(13)  # start the same way each time...

        for section in self.cfg_parser.sections():
            config = ConfigSection(self.cfg_parser[section])
            logging.info("***** Parsing section {} *****".format(section))
            if section.startswith("convolutional"):
                convolutional(config, self.container)

            elif section.startswith("InvResidual"):
                invResidual(config, self.container)

            elif section.startswith("Linear"):
                linear(config, self.container)

            elif section.startswith("globalaveragepool"):
                globalaveragepool(config, self.container)

            elif section.startswith("input"):
                input(config, self.container)

            elif section.startswith("output"):
                output(config, self.container)
            elif section.startswith("net"):
                pass

            else:
                raise ValueError("Unsupported section header type: {}".format(section))

        # Create and save model.
        logging.info("done reading config file")
        # assume the end of the network is an output if none are define.
        if len(self.container.out_index) == 0:
            logging.info(
                "No outputs defined, so we are assuming last layer is the output and define it as such"
            )
            self.container.out_index.append(len(self.container.all_layers) - 1)
        model = Model(
            inputs=[self.container.all_layers[i] for i in self.container.in_index],
            outputs=[self.container.all_layers[i] for i in self.container.out_index],
        )
        logging.info("done assembling model")
        model.summary(print_fn=print)

        # logging.info() all inputs, formatted for use with coremltools Keras convertor
        print("input_names = [")
        for name in self.container.in_names:
            print(f"    '{name}',")
        print("],")

        # logging.info() all outputs, formatted for use with coremltools Keras convertor
        print("output_names = [")
        for name in self.container.out_names:
            print(f"    '{name}',")
        print("],")

        # Just for fun, logging.info() all inputs and outputs and their shapes.

        # Inputs are actual Keras layers so we extract their names
        # also build input_features for CoreML generation
        for layer in [self.container.all_layers[i] for i in self.container.in_index]:
            logging.info(f"name: {layer.name}; shape: {layer.shape}")

        # Outputs are not Keras layers, so we have a separate list of names for them
        # - name comes from our list,
        # - shape comes from the Keras layer they point to
        out_layers = [self.container.all_layers[i] for i in self.container.out_index]
        for i in range(len(self.container.out_names)):
            logging.info(
                f"name: {self.container.out_names[i]}; shape: {out_layers[i].shape}"
            )
        return (
            model,
            self.container.fake_weights,
            self.container.in_names,
            self.container.out_names,
            self.container.image_inputs,
        )
