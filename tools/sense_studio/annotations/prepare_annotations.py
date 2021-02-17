#!/usr/bin/env python

import os
import urllib

from flask import Blueprint
from flask import redirect
from flask import url_for
from os.path import join

from sense.finetuning import compute_frames_features
from tools.sense_studio.utils import _load_feature_extractor
from tools.sense_studio.utils import SPLITS

prepare_annotations_bp = Blueprint('prepare_annotations_bp', __name__, template_folder='templates/annotations')


@prepare_annotations_bp.route('/<path:path>')
def prepare_annotation(path):
    """
    Prepare all files needed for annotating the videos in the given project.
    """

    path = f'/{urllib.parse.unquote(path)}'  # Make path absolute
    # load feature extractor if needed
    inference_engine = _load_feature_extractor()
    for split in SPLITS:
        print("\n" + "-" * 10 + f"Preparing videos in the {split}-set" + "-" * 10)
        for label in os.listdir(join(path, f'videos_{split}')):
            compute_frames_features(inference_engine, split, label, path)
    return redirect(url_for("project_details", path=path))
