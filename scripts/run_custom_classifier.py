#!/usr/bin/env python
"""
Run a custom classifier that was obtained via the train_classifier script.

Usage:
  run_custom_classifier.py --custom_classifier=PATH
                           [--camera_id=CAMERA_ID]
                           [--path_in=FILENAME]
                           [--path_out=FILENAME]
                           [--title=TITLE]
                           [--use_gpu]
  run_custom_classifier.py (-h | --help)

Options:
  --custom_classifier=PATH   Path to the custom classifier to use
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
"""
import os
import json
from docopt import docopt

import realtimenet.display
from realtimenet import camera
from realtimenet import engine
from realtimenet import feature_extractors
from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet.downstream_tasks.postprocess import PostprocessClassificationOutput


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    camera_id = args['--camera_id'] or 0
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    custom_classifier = args['--custom_classifier'] or None
    title = args['--title'] or None
    use_gpu = args['--use_gpu']

    # Load original feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    checkpoint = engine.load_weights('resources/strided_inflated_efficientnet.ckpt')

    # Load custom classifier
    checkpoint_classifier = engine.load_weights(os.path.join(custom_classifier, 'classifier.checkpoint'))
    # Update original weights in case some intermediate layers have been finetuned
    name_finetuned_layers = set(checkpoint.keys()).intersection(checkpoint_classifier.keys())
    for key in name_finetuned_layers:
        checkpoint[key] = checkpoint_classifier.pop(key)

    with open(os.path.join(custom_classifier, 'label2int.json')) as file:
        class2int = json.load(file)
    INT2LAB = {value: key for key, value in class2int.items()}

    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=len(INT2LAB))
    gesture_classifier.load_state_dict(checkpoint_classifier)
    gesture_classifier.eval()

    # Concatenate feature extractor and met converter
    net = Pipe(feature_extractor, gesture_classifier)

    # Create inference engine, video streaming and display instances
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)

    video_source = camera.VideoSource(camera_id=camera_id,
                                      size=inference_engine.expected_frame_size,
                                      filename=path_in)

    framegrabber = camera.VideoStream(video_source,
                                      inference_engine.fps)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4)
    ]

    display_ops = [
        realtimenet.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0.1),
    ]
    display_results = realtimenet.display.DisplayResults(title=title, display_ops=display_ops)

    engine.run_inference_engine(inference_engine,
                                framegrabber,
                                postprocessor,
                                display_results,
                                path_out)
