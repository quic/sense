#! /usr/bin/env python
"""
Pytorch-to-Tflite conversion script.
Author: Mark Todorovich.


Usage:
  convert_to_tflite.py --classifier=CLASSIFIER --output_name=OUTPUT_NAME
                       [--backbone_name=BACKBONE_NAME]
                       [--backbone_version=BACKBONE_VERSION]
                       [--path_in=PATH]
                       [--plot_model]
                       [--float32]
                       [--verbose]
  convert_to_tflite.py (-h | --help)

Options:
  --classifier=CLASSIFIER              Name of the classifier model. Either one of the pre-trained classifiers
                                       such as "gesture_recognition" or "fitness_activity_recognition",
                                       or "custom_classifier" for converting a custom trained one.
                                       For the pre-trained classifiers, a name and version for the backbone model
                                       need to be provided. For a custom classifier, path_in needs to be provided.
  --output_name=OUTPUT_NAME            Base name of the output file.
  --backbone_name=BACKBONE_NAME        Name of the backbone model, e.g. "StridedInflatedEfficientNet"
  --backbone_version=BACKBONE_VERSION  Version of the backbone model, e.g. "pro" or "lite"
  --path_in=PATH                       Path to the trained classifier directory if converting custom classifier.
  --plot_model                         Plot intermediate Keras model and save as image.
  --float32                            Use full precision. By default, this script quantizes weights
                                       to 16-bit precision.
  --verbose                            Enable detailed logging

  -h --help
"""

import os
import logging
from docopt import docopt
from keras.utils.vis_utils import plot_model as plot

from sense import RESOURCES_DIR
from sense.loading import ModelConfig
from tools.conversion.config_loader import finalize_custom_classifier_config
from tools.conversion.config_loader import load_config
from tools.conversion.keras_converter import KerasConverter
from tools.conversion.keras_exporter import export_keras_to_tflite
from tools.conversion.weights_loader import load_custom_classifier_weights


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

EFFICIENTNET = 'StridedInflatedEfficientNet'

SUPPORTED_BACKBONE_CONVERSIONS = {
    EFFICIENTNET: {
        "config_file": "tools/conversion/cfg/efficientnet.cfg",
        "conversion_parameters": {
            **DEFAULT_CONVERSION_PARAMETERS,
            "image_scale": 255.0,
        },
    }
}

SUPPORTED_CLASSIFIER_CONVERSIONS = {
    "gesture_recognition": {
        "config_file": "tools/conversion/cfg/logistic_regression.cfg",
        "placeholder_values": {"NUM_CLASSES": "30"},
    },
    "fitness_activity_recognition": {
        "config_file": "tools/conversion/cfg/logistic_regression.cfg",
        "placeholder_values": {"NUM_CLASSES": "81"},
    },
    "custom_classifier": {
        "config_file": "tools/conversion/cfg/logistic_regression.cfg",
        "placeholder_values": {"NUM_CLASSES": None},
    },
}


def convert(backbone_settings, classifier_settings, weights_full, output_name, plot_model):
    output_dir = os.path.join(RESOURCES_DIR, "model_conversion")
    os.makedirs(output_dir, exist_ok=True)

    conversion_parameters = backbone_settings["conversion_parameters"]
    keras_file = os.path.join(output_dir, output_name + ".h5")
    tflite_file = os.path.join(output_dir, output_name + ".tflite")

    cfg_parser = load_config(backbone_settings, classifier_settings)
    keras_converter = KerasConverter(cfg_parser, weights_full, conversion_parameters)

    (
        model,
        fake_weights,
        in_names,
        out_names,
        image_inputs,
    ) = keras_converter.create_keras_model()

    model.save(keras_file)
    logging.info(f"Saved Keras model to {keras_file}")

    if plot_model:
        to_file = os.path.join(output_dir, output_name + ".png")
        plot(model, to_file=to_file, show_shapes=True)
        logging.info(f"Saved model plot to {to_file}")

    logging.info(f"input_names {in_names}")
    logging.info(f"output_names {out_names}")
    logging.info(f"image_input_names {image_inputs}")
    logging.info(f"keras file type: {type(keras_file)}")

    export_keras_to_tflite(keras_file, tflite_file)

    if fake_weights:
        logging.error(
            "************************* Warning!! ***************************\n"
            "Weights in checkpoint did not match weights required by network\n"
            "Fake weights were generated where they were needed!!!!!!!!!!!!!\n"
            "************************* Warning!! ***************************"
        )


if __name__ == "__main__":
    args = docopt(__doc__)
    classifier_name = args["--classifier"]
    output_name = args["--output_name"]
    backbone_name = args["--backbone_name"]
    backbone_version = args["--backbone_version"]
    path_in = args["--path_in"]
    plot_model = args["--plot_model"]
    float32 = args["--float32"]
    verbose = args["--verbose"]

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    classifier_settings = SUPPORTED_CLASSIFIER_CONVERSIONS.get(classifier_name)
    if not classifier_settings:
        raise Exception(
            f"Classifier not found: {classifier_name}. Only the following classifiers "
            f"can be converted: {list(SUPPORTED_CLASSIFIER_CONVERSIONS.keys())}"
        )

    if classifier_name == "custom_classifier":
        if not path_in:
            raise ValueError("You have to provide the directory used to train the custom classifier")

        backbone_model_config, weights = load_custom_classifier_weights(path_in)
        backbone_name = backbone_model_config.model_name
        finalize_custom_classifier_config(classifier_settings, path_in)
    else:
        if not backbone_name or not backbone_version:
            raise ValueError("You have to provide the name and version for the backbone model")

        model_config = ModelConfig(backbone_name, backbone_version, [classifier_name])
        weights = model_config.load_weights()

    backbone_settings = SUPPORTED_BACKBONE_CONVERSIONS.get(backbone_name)
    if not backbone_settings:
        raise Exception(
            f"Backbone not found: {backbone_name}. Only the following backbones "
            f"can be converted: {list(SUPPORTED_BACKBONE_CONVERSIONS.keys())}"
        )

    # Merge weights (possibly overwriting backbone weights with finetuned ones from classifier checkpoint)
    weights_full = weights['backbone']
    weights_full.update(weights[classifier_name])

    for key, weight in weights_full.items():
        logging.info(f"{key}: {weight.shape}")

    convert(
        backbone_settings,
        classifier_settings,
        weights_full,
        output_name,
        plot_model,
    )
