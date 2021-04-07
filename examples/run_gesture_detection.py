#!/usr/bin/env python
"""
Real time detection of 6 hand gestures.

Usage:
  run_gesture_detection.py [--camera_id=CAMERA_ID]
                           [--path_in=FILENAME]
                           [--path_out=FILENAME]
                           [--title=TITLE]
                           [--model_name=NAME]
                           [--model_version=VERSION]
                           [--use_gpu]
  run_gesture_detection.py (-h | --help)

Options:
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
  --model_name=NAME          Name of the model to be used.
  --model_version=VERSION    Version of the model to be used.
  --use_gpu                  Whether to run inference on the GPU or not.
"""
from docopt import docopt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks.gesture_detection import INT2LAB
from sense.downstream_tasks.gesture_detection import LAB_THRESHOLDS
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.loading import get_relevant_weights
from sense.loading import build_backbone_network
from sense.loading import ModelConfig


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['gesture_detection']),
    ModelConfig('StridedInflatedEfficientNet', 'lite', ['gesture_detection']),
]

if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    camera_id = int(args['--camera_id'] or 0)
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    title = args['--title'] or None
    model_name = args['--model_name'] or None
    model_version = args['--model_version'] or None
    use_gpu = args['--use_gpu']

    # Load weights
    selected_config, weights = get_relevant_weights(
        SUPPORTED_MODEL_CONFIGURATIONS,
        model_name,
        model_version
    )

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'],
                                              weights_finetuned=weights['gesture_detection'])

    # Create a logistic regression classifier
    gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                            num_out=len(INT2LAB))
    gesture_classifier.load_state_dict(weights['gesture_detection'], strict=False)
    gesture_classifier.eval()

    # Concatenate backbone network and logistic regression
    net = Pipe(backbone_network, gesture_classifier)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=1)
    ]

    border_size = 30

    display_ops = [
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        sense.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0., x_offset=500),
        sense.display.DisplayClassnameOverlay(thresholds=LAB_THRESHOLDS,
                                              border_size=border_size if not title else border_size + 50,
                                              duration=1),
    ]
    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops)

    # Run live inference
    controller = Controller(
        neural_network=net,
        post_processors=postprocessor,
        results_display=display_results,
        callbacks=[],
        camera_id=camera_id,
        path_in=path_in,
        path_out=path_out,
        use_gpu=use_gpu
    )
    controller.run_inference()
