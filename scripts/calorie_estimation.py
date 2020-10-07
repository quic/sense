#!/usr/bin/env python
"""
Live estimation of burned calories.

Usage:
  calorie_estimation.py [--weight=WEIGHT --age=AGE --height=HEIGHT --gender=GENDER]
                        [--camera_id=CAMERA_ID]
                        [--path_in=FILENAME]
                        [--path_out=FILENAME]
                        [--title=TITLE]
                        [--use_gpu]
  calorie_estimation.py (-h | --help)

Options:
  --weight=WEIGHT                 Weight (in kilograms). Will be used to convert predicted MET value to calories [default: 70]
  --age=AGE                       Age (in years). Will be used to convert predicted MET value to calories [default: 30]
  --height=HEIGHT                 Height (in centimeters). Will be used to convert predicted MET value to calories [default: 170]
  --gender=GENDER                 Gender ("male" or "female" or "other"). Will be used to convert predicted MET value to calories
  --camera_id=CAMERA_ID           ID of the camera to stream from
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
from realtimenet.downstream_tasks.nn_utils import Pipe

if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    weight = float(args['--weight'])
    height = float(args['--height'])
    age = float(args['--age'])
    gender = args['--gender'] or None
    use_gpu = args['--use_gpu']

    camera_id = args['--camera_id'] or 0
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    title = args['--title'] or None

    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedMobileNetV2()
    checkpoint = engine.load_weights('resources/strided_inflated_mobilenet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Load MET value converter
    met_value_converter = calorie_estimation.METValueMLPConverter()
    checkpoint = engine.load_weights('resources/calorie_estimation/mobilenet_features_met_converter.ckpt')
    met_value_converter.load_state_dict(checkpoint)
    met_value_converter.eval()

    # Concatenate feature extractor and met converter
    net = Pipe(feature_extractor, met_value_converter)

    # Create inference engine, video streaming and display objects
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)

    video_source = camera.VideoSource(camera_id=camera_id,
                                      size=inference_engine.expected_frame_size,
                                      filename=path_in)

    framegrabber = camera.VideoStream(video_source,
                                      inference_engine.fps)

    post_processors = [
        calorie_estimation.CalorieAccumulator(weight=weight,
                                              height=height,
                                              age=age,
                                              gender=gender,
                                              smoothing=12)
    ]

    display_ops = [
        realtimenet.display.DisplayDetailedMETandCalories(),
    ]
    display_results = realtimenet.display.DisplayResults(title=title, display_ops=display_ops)

    # Run live inference
    engine.run_inference_engine(inference_engine,
                                framegrabber,
                                post_processors,
                                display_results,
                                path_out)
