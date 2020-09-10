#!/usr/bin/env python
"""
Tracks which fitness exercises is performed and estimates total number of calories.

Usage:
  fitness_tracker.py [--weight=WEIGHT --age=AGE --height=HEIGHT --gender=GENDER]
                     [--camera_id=CAMERA_ID]
                     [--path_in=FILENAME]
                     [--path_out=FILENAME]
                     [--title=TITLE]
                     [--use_gpu]
  fitness_tracker.py (-h | --help)

Options:
  --weight=WEIGHT                 Weight (in kilograms). Will be used to convert predicted MET value to calories [default: 70]
  --age=AGE                       Age (in years). Will be used to convert predicted MET value to calories [default: 30]
  --height=HEIGHT                 Height (in centimeters). Will be used to convert predicted MET value to calories [default: 170]
  --gender=GENDER                 Gender ("male" or "female" or "other"). Will be used to convert predicted MET value to calories
  --path_in=FILENAME              Video file to stream from
  --path_out=FILENAME             Video file to stream to
  --title=TITLE                   This adds a title to the window display
"""
import torch
from docopt import docopt

import realtimenet.display
from realtimenet import camera
from realtimenet import engine
from realtimenet import feature_extractors
from realtimenet.downstream_tasks import calorie_estimation
from realtimenet.downstream_tasks.fitness_activity_recognition import INT2LAB
from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet.downstream_tasks.postprocess import PostprocessClassificationOutput


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    weight = float(args['--weight'])
    height = float(args['--height'])
    age = float(args['--age'])
    gender = args['--gender'] or None
    camera_id = args['--camera_id'] or 0
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    title = args['--title'] or None
    use_gpu = args['--use_gpu']

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedMobileNetV2()
    checkpoint = torch.load('resources/strided_inflated_mobilenet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Load fitness activity classifier
    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=81)
    checkpoint = torch.load('resources/fitness_activity_recognition/mobilenet_logistic_regression.ckpt')
    gesture_classifier.load_state_dict(checkpoint)
    gesture_classifier.eval()

    # Load MET value converter
    met_value_converter = calorie_estimation.METValueMLPConverter()
    checkpoint = torch.load('resources/calorie_estimation/mobilenet_features_met_converter.ckpt')
    met_value_converter.load_state_dict(checkpoint)
    met_value_converter.eval()

    # Concatenate feature extractor with downstream nets
    net = Pipe(feature_extractor, feature_converter=[gesture_classifier,
                                                     met_value_converter])

    # Create inference engine, video streaming and display objects
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)

    video_source = camera.VideoSource(camera_id=camera_id,
                                      size=inference_engine.expected_frame_size,
                                      filename=path_in)

    framegrabber = camera.VideoStream(video_source,
                                      inference_engine.fps)

    post_processors = [
        PostprocessClassificationOutput(INT2LAB, smoothing=8,
                                        indices=[0]),
        calorie_estimation.CalorieAccumulator(weight=weight,
                                              height=height,
                                              age=age,
                                              gender=gender,
                                              smoothing=12,
                                              indices=[1])
    ]

    display_ops = [
        realtimenet.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0.5, y_offset=0),
        realtimenet.display.DisplayMETandCalories(y_offset=50),
    ]
    display_results = realtimenet.display.DisplayResults(title=title,
                                                         display_ops=display_ops,
                                                         border_size=70)

    # Run live inference
    engine.run_inference_engine(inference_engine,
                                framegrabber,
                                post_processors,
                                display_results,
                                path_out)
