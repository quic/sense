#!/usr/bin/env python
"""
Real time detection of 30 actions.

Usage:
  run_action_recognition.py [--camera_id=CAMERA_ID]
                            [--path_in=FILENAME]
                            [--path_out=FILENAME]
                            [--title=TITLE]
                            [--model_name=NAME]
                            [--model_version=VERSION]
                            [--use_gpu]
  run_action_recognition.py (-h | --help)

Options:
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
  --model_name=NAME          Name of the model to be used.
  --model_version=VERSION    Version of the model to be used.
  --use_gpu                  Whether to run inference on the GPU or not.
"""
from typing import Optional

from docopt import docopt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks.action_recognition import INT2LAB
from sense.downstream_tasks.action_recognition import LAB_THRESHOLDS
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.loading import get_relevant_weights
from sense.loading import build_backbone_network
from sense.loading import ModelConfig


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['action_recognition']),
    ModelConfig('StridedInflatedMobileNetV2', 'pro', ['action_recognition']),
    ModelConfig('StridedInflatedEfficientNet', 'lite', ['action_recognition']),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', ['action_recognition']),
]


def run_action_recognition(model_name: str, model_version: str, path_in: Optional[str] = None,
                           path_out: Optional[str] = None, title: Optional[str] = None, camera_id: int = 0,
                           use_gpu: bool = True, **kwargs):
    """
    :param model_name:
        Model from backbone (StridedInflatedEfficientNet or StridedInflatedMobileNetV2).
    :param model_version:
        Model version (pro or lite)
    :param path_in:
        The index of the webcam that is used. Default to 0.
    :param path_out:
        If provided, store the captured video in a file in this location.
    :param title:
            Title of the image frame on display.
    :param camera_id:
        The index of the webcam that is used. Default id is 0.
    :param use_gpu:
        If True, run the model on the GPU
    """
    # Load weights
    selected_config, weights = get_relevant_weights(
        SUPPORTED_MODEL_CONFIGURATIONS,
        model_name,
        model_version
    )

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'])

    # Create a logistic regression classifier
    action_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                           num_out=30)
    action_classifier.load_state_dict(weights['action_recognition'])
    action_classifier.eval()

    # Concatenate backbone network and logistic regression
    net = Pipe(backbone_network, action_classifier)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4)
    ]

    border_size = 30

    display_ops = [
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        sense.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0.5),
        sense.display.DisplayClassnameOverlay(thresholds=LAB_THRESHOLDS,
                                              border_size_top=border_size if not title else border_size + 50),
    ]
    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops,
                                                   display_fn=kwargs.get('display_fn', None))

    # Run live inference
    controller = Controller(
        neural_network=net,
        post_processors=postprocessor,
        results_display=display_results,
        callbacks=[],
        camera_id=camera_id,
        path_in=path_in,
        path_out=path_out,
        use_gpu=use_gpu,
        stop_event=kwargs.get('stop_event', None),
    )
    controller.run_inference()


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    _camera_id = int(args['--camera_id'] or 0)
    _path_in = args['--path_in'] or None
    _path_out = args['--path_out'] or None
    _title = args['--title'] or None
    _model_name = args['--model_name'] or None
    _model_version = args['--model_version'] or None
    _use_gpu = args['--use_gpu']

    run_action_recognition(
        model_name=_model_name,
        model_version=_model_version,
        path_in=_path_in,
        path_out=_path_out,
        title=_title,
        camera_id=0,
        use_gpu=_use_gpu,
    )
