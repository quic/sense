import configparser
import io
from collections import defaultdict
import json
import numpy as np
import os


def merge_backbone_and_classifier_cfg_files(
    backbone_config_file, classifier_config_file, placeholder_values=None
):
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
                if line.startswith("["):
                    section = line.strip().strip("[]")
                    _section = section + "_" + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)

                for key, value in placeholder_values.items():
                    line = line.replace(key, value)

                output_stream.write(line)

    output_stream.seek(0)
    return output_stream


def load_config(backbone_settings, classifier_settings):
    print("Parsing CFG file.")
    placeholder_values = {
        **backbone_settings.get("placeholder_values", {}),
        **classifier_settings.get("placeholder_values", {}),
    }
    unique_config_file = merge_backbone_and_classifier_cfg_files(
        backbone_settings["config_file"],
        classifier_settings["config_file"],
        placeholder_values=placeholder_values,
    )
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    return cfg_parser


def finalize_custom_classifier_config(classifier_settings, path_in, backbone_name):
    # if custom classifier, fill the classifier settings with arguments
    if not path_in:
        raise Exception(
            "You have to provide the directory used to train the custom classifier"
        )

    weights_file = os.path.join(path_in, "classifier.checkpoint")
    if not os.path.isfile(weights_file):
        raise Exception(
            f'Missing weights: "classifier.checkpoint" file was not found in {path_in}'
        )

    lab2int_file = os.path.join(path_in, "label2int.json")
    if not os.path.isfile(lab2int_file):
        raise Exception(
            f'Missing label mapping: "label2int.json" file was not found in {path_in}'
        )
    try:
        with open(lab2int_file, "r") as f:
            num_classes = np.max(list(json.load(f).values())) + 1
    except:
        raise Exception(f'Error while parsing "label2int.json"')

    classifier_settings["corresponding_backbone"] = backbone_name
    classifier_settings["weights_file"] = weights_file
    classifier_settings["placeholder_values"]["NUM_CLASSES"] = str(num_classes)

    return classifier_settings
