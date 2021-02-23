import logging

from sense.loading import load_backbone_weights, load_weights_from_resources


def load_weights(backbone_ckpt, classifier_ckpt):
    # Load weights and config.
    logging.info("Loading weights.")
    weights_backbone = load_backbone_weights(backbone_ckpt)
    weights_classifier = load_weights_from_resources(classifier_ckpt)
    # if some deeper layer have been finetuned, change them in the backbone weights dictionary
    name_finetuned_layers = set(weights_backbone.keys()).intersection(
        weights_classifier.keys()
    )
    for key in name_finetuned_layers:
        weights_backbone[key] = weights_classifier.pop(key)
    weights_full = {**weights_backbone, **weights_classifier}

    for key in weights_full.keys():
        logging.info(f"{key}: {weights_full[key].shape}")

    return weights_full
