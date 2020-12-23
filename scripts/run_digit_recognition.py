#!/usr/bin/env python
"""
Real time recognition of digits drawn in the air by hand.

Usage:
  run_digit_recognition.py [--camera_id=CAMERA_ID]
                           [--path_in=FILENAME]
                           [--path_out=FILENAME]
                           [--title=TITLE]
                           [--use_gpu]
  run_digit_recognition.py (-h | --help)

Options:
  --camera_id=CAMERA_ID      Index of the camera to be used as input. Defaults to 0.
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
  --use_gpu                  Use GPU for inference
"""
from docopt import docopt

import sense.display
from sense import engine
from sense import feature_extractors
from sense.controller import Controller
from sense.downstream_tasks.digit_recognition import INT2LAB, LAB2THRESHOLD
from sense.downstream_tasks.nn_utils import Pipe, LogisticRegression
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput


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
    digit_classifier = LogisticRegression(num_in=feature_extractor.feature_dim, num_out=len(INT2LAB))
    checkpoint = engine.load_weights('resources/digit_recognition/efficientnet_logistic_regression.ckpt')
    digit_classifier.load_state_dict(checkpoint)
    digit_classifier.eval()

    # Concatenate feature extractor and digit classifier
    net = Pipe(feature_extractor, digit_classifier)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4)
    ]

    border_size = 50  # Increase border size for showing top 2 preditions

    display_ops = [
        sense.display.DisplayTopKClassificationOutputs(top_k=2, threshold=0),
        sense.display.DisplayDigits(thresholds=LAB2THRESHOLD,
                                    duration=2,
                                    border_size=border_size if not title else border_size + 50),
    ]
    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops, border_size=border_size)

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
