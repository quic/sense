import torch

def load_weights(backbone_ckpt, classifier_ckpt):
    # Load weights and config.
    print("Loading weights.")
    weights_backbone = torch.load(backbone_ckpt, map_location="cpu")
    weights_classifier = torch.load(classifier_ckpt, map_location="cpu")
    # if some deeper layer have been finetuned, change them in the backbone weights dictionary
    name_finetuned_layers = set(weights_backbone.keys()).intersection(
        weights_classifier.keys()
    )
    for key in name_finetuned_layers:
        weights_backbone[key] = weights_classifier.pop(key)
    weights_full = {**weights_backbone, **weights_classifier}

    for key in weights_full.keys():
        print(key, weights_full[key].shape)

    return weights_full