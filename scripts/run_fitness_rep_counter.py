#!/usr/bin/env python
"""
Real-time rep counter for jumping jacks and squats.

Usage:
  run_fitness_rep_counter.py [--camera_id=CAMERA_ID]
                             [--path_in=FILENAME]
                             [--path_out=FILENAME]
                             [--title=TITLE]
                             [--use_gpu]
  run_fitness_rep_counter.py (-h | --help)

Options:
  --path_in=FILENAME              Video file to stream from
  --path_out=FILENAME             Video file to stream to
  --title=TITLE                   This adds a title to the window display
"""
from docopt import docopt

import realtimenet.display
from realtimenet import camera
from realtimenet import engine
from realtimenet import feature_extractors
from realtimenet.controller import Controller
from realtimenet.downstream_tasks.fitness_rep_counting import INT2LAB
from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet.downstream_tasks.postprocess import PostprocessRepCounts, PostprocessClassificationOutput


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    camera_id = args['--camera_id'] or 0
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    title = args['--title'] or None
    use_gpu = args['--use_gpu']

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    checkpoint = engine.load_weights('resources/backbone/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Load a logistic regression classifier
    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=5)
    checkpoint = engine.load_weights('resources/fitness_rep_counting/efficientnet_logistic_regression.ckpt')
    gesture_classifier.load_state_dict(checkpoint)
    gesture_classifier.eval()

    # Concatenate feature extractor and met converter
    net = Pipe(feature_extractor, gesture_classifier)

    postprocessor = [
        PostprocessRepCounts(INT2LAB),
        PostprocessClassificationOutput(INT2LAB, smoothing=1)
    ]

    display_ops = [
        realtimenet.display.DisplayFPS(expected_camera_fps=net.fps, expected_inference_fps=net.fps/net.step_size,
                                       y_offset=115),
        realtimenet.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0.5),
        realtimenet.display.DisplayRepCounts()
    ]
    display_results = realtimenet.display.DisplayResults(title=title, display_ops=display_ops,
                                                         border_size=100)

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
