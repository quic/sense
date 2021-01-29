#! /usr/bin/env python
"""
Pytorch-to-Tflite conversion script.
Author: Mark Todorovich.


Usage:
  convert_to_tflite.py --backbone=NAME --classifier=NAME --output_name=NAME
                       [--path_in=PATH]
                       [--plot_model]
                       [--float32]
  convert_to_tflite.py (-h | --help)

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

import os
import logging
from docopt import docopt
from keras.utils.vis_utils import plot_model as plot

from tools.conversion.config_loader import load_config
from tools.conversion.config_loader import finalize_custom_classifier_config
from tools.conversion.weights_loader import load_weights
from tools.conversion.keras_converter import KerasConverter
from tools.conversion.keras_exporter import export_keras_to_tflite


DEFAULT_CONVERSION_PARAMETERS = {
    "image_scale": 1.0,
    "normalize_inputs": False,
    "red_bias": None,
    "green_bias": None,
    "blue_bias": None,
    "red_scale": None,
    "green_scale": None,
    "blue_scale": None,
    "use_prelu": False,
}

SUPPORTED_BACKBONE_CONVERSIONS = {
    "efficientnet": {
        "config_file": "tools/conversion/cfg/efficientnet.cfg",
        "weights_file": "resources/backbone/strided_inflated_efficientnet.ckpt",
        "conversion_parameters": {
            **DEFAULT_CONVERSION_PARAMETERS,
            "image_scale": 255.0,
        },
    }
}

SUPPORTED_CLASSIFIER_CONVERSIONS = {
    "efficient_net_gesture_control": {
        "config_file": "tools/conversion/cfg/logistic_regression.cfg",
        "placeholder_values": {"NUM_CLASSES": "30"},
        "weights_file": "resources/gesture_detection/efficientnet_logistic_regression.ckpt",
        "corresponding_backbone": "efficientnet",
    },
    "efficient_net_fitness_activity_recognition": {
        "config_file": "tools/conversion/cfg/logistic_regression.cfg",
        "placeholder_values": {"NUM_CLASSES": "81"},
        "weights_file": "resources/fitness_activity_recognition/efficientnet_logistic_regression.ckpt",
        "corresponding_backbone": "efficientnet",
    },
    "custom_classifier": {
        "config_file": "tools/conversion/cfg/logistic_regression.cfg",
        "placeholder_values": {"NUM_CLASSES": None},
        "weights_file": None,
        "corresponding_backbone": None,
    },
}


def convert(backbone_settings, classifier_settings, output_name, plot_model):
    output_dir = "resources/model_conversion/"
    os.makedirs(output_dir, exist_ok=True)

    conversion_parameters = backbone_settings["conversion_parameters"]
    keras_file = os.path.join(output_dir, output_name + ".h5")
    tflite_file = os.path.join(output_dir, output_name + ".tflite")

    weights_full = load_weights(
        backbone_settings["weights_file"], classifier_settings["weights_file"]
    )

    cfg_parser = load_config(backbone_settings, classifier_settings)
    keras_converter = KerasConverter(cfg_parser, weights_full, conversion_parameters)

    (
        model,
        fake_weights,
        in_names,
        out_names,
        image_inputs,
    ) = keras_converter.create_keras_model()

    model.save("{}".format(keras_file))
    logging.info("Saved Keras model to {}".format(keras_file))

    if plot_model:
        to_file = os.path.join(output_dir, output_name + ".png")
        plot(model, to_file=to_file, show_shapes=True)
        logging.info("Saved model plot to {}".format(to_file))

    logging.info("input_names", in_names)
    logging.info("output_names", out_names)
    logging.info("image_input_names", image_inputs)
    logging.info(type(keras_file))

    export_keras_to_tflite(keras_file, tflite_file)

    if fake_weights:
        logging.warning("************************* Warning!! **************************")
        logging.warning("Weights in checkpoint did not match weights required by network")
        logging.warning("Fake weights were generated where they were needed!!!!!!!!!!!!")
        logging.warning("************************* Warning!! **************************")


if __name__ == "__main__":
    args = docopt(__doc__)
    backbone_name = args["--backbone"]
    classifier_name = args["--classifier"]
    output_name = args["--output_name"]
    float32 = args["--float32"]
    path_in = args["--path_in"]
    plot_model = args["--plot_model"]

    backbone_settings = SUPPORTED_BACKBONE_CONVERSIONS.get(backbone_name)
    if not backbone_settings:
        raise Exception(
            "Backbone not found: {}. Only the following backbones "
            "can be converted: {}".format(
                backbone_name, SUPPORTED_BACKBONE_CONVERSIONS.keys()
            )
        )

    classifier_settings = SUPPORTED_CLASSIFIER_CONVERSIONS.get(classifier_name)
    if not classifier_settings:
        raise Exception(
            "Classifier not found: {}. Only the following backbones "
            "can be converted: {}".format(
                classifier_name, SUPPORTED_CLASSIFIER_CONVERSIONS.keys()
            )
        )
    if classifier_name == "custom_classifier":
        classifier_settings = finalize_custom_classifier_config(
            classifier_settings, path_in, backbone_name
        )

    if classifier_settings["corresponding_backbone"] != backbone_name:
        raise Exception(
            "This classifier expects a different backbone: "
            "{}".format(classifier_settings["corresponding_backbone"])
        )

    convert(backbone_settings, classifier_settings, output_name, plot_model)
