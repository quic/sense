import configparser
from collections import defaultdict
import json
import numpy as np
import os
import logging


def merge_backbone_and_classifier_cfg_files(
    backbone_config_file, classifier_config_file, placeholder_values=None
):
    """
    Concatenate backbone and classifier config files and make sure all config sections
    have unique names (adding unique suffixes) for compatibility with configparser.
    """
    placeholder_values = placeholder_values or {}
    section_counters = defaultdict(int)

    output = ''

    for cfg_file in [backbone_config_file, classifier_config_file]:

        with open(cfg_file) as fp:
            lines = fp.readlines()

        for line in lines:
            # Make sure section names are unique
            if line.startswith("["):
                section = line.strip().strip("[]")
                _section = f"{section}_{str(section_counters[section])}"
                section_counters[section] += 1
                line = line.replace(section, _section)

            for key, value in placeholder_values.items():
                line = line.replace(key, value)

            output += line

    return output


def load_config(backbone_settings, classifier_settings):
    logging.info("Parsing CFG file.")
    placeholder_values = {
        **backbone_settings.get("placeholder_values", {}),
        **classifier_settings.get("placeholder_values", {}),
    }
    unique_config = merge_backbone_and_classifier_cfg_files(
        backbone_settings["config_file"],
        classifier_settings["config_file"],
        placeholder_values=placeholder_values,
    )
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_string(unique_config)
    return cfg_parser


def finalize_custom_classifier_config(classifier_settings, path_in):
    # Infer number of classes from label2int file
    lab2int_file = os.path.join(path_in, "label2int.json")
    with open(lab2int_file, "r") as f:
        num_classes = np.max(list(json.load(f).values())) + 1

    classifier_settings["placeholder_values"]["NUM_CLASSES"] = str(num_classes)
