#!/usr/bin/env python
"""
Tracks which fitness exercises is performed and estimates total number of calories.

Usage:
  run_fitness_tracker.py [--weight=WEIGHT --age=AGE --height=HEIGHT --gender=GENDER]
                         [--camera_id=CAMERA_ID]
                         [--path_in=FILENAME]
                         [--path_out=FILENAME]
                         [--title=TITLE]
                         [--model_name=NAME]
                         [--model_version=VERSION]
                         [--use_gpu]
  run_fitness_tracker.py (-h | --help)

Options:
  --weight=WEIGHT          Weight (in kilograms). Will be used to convert MET values to calories [default: 70]
  --age=AGE                Age (in years). Will be used to convert MET values to calories [default: 30]
  --height=HEIGHT          Height (in centimeters). Will be used to convert MET values to calories [default: 170]
  --gender=GENDER          Gender ("male" or "female" or "other"). Will be used to convert MET values to calories
  --path_in=FILENAME       Video file to stream from
  --path_out=FILENAME      Video file to stream to
  --title=TITLE            This adds a title to the window display
  --model_name=NAME        Name of the model to be used.
  --model_version=VERSION  Version of the model to be used.
  --use_gpu                Whether to run inference on the GPU or not.
"""
from docopt import docopt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks import calorie_estimation
from sense.downstream_tasks.fitness_activity_recognition import INT2LAB
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig


REQUIRED_FEATURE_CONVERTERS = ['fitness_activity_recognition', 'met_converter']

SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedMobileNetV2', 'pro', REQUIRED_FEATURE_CONVERTERS),
    ModelConfig('StridedInflatedEfficientNet', 'pro', REQUIRED_FEATURE_CONVERTERS),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', REQUIRED_FEATURE_CONVERTERS),
    ModelConfig('StridedInflatedEfficientNet', 'lite', REQUIRED_FEATURE_CONVERTERS),
]


def run_fitness_tracker(model_name, model_version, path_in=None, path_out=None, weight=70.0, height=170.0, age=30.0,
                        gender=None, title=None, camera_id=0, use_gpu=True, display_fn=None, stop_event=None, **kwargs):
    # Load weights
    selected_config, weights = get_relevant_weights(
        SUPPORTED_MODEL_CONFIGURATIONS,
        model_name,
        model_version
    )

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'])

    # Create fitness activity classifier
    gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                            num_out=81)
    gesture_classifier.load_state_dict(weights['fitness_activity_recognition'])
    gesture_classifier.eval()

    # Create MET value converter
    met_value_converter = calorie_estimation.METValueMLPConverter()
    met_value_converter.load_state_dict(weights['met_converter'])
    met_value_converter.eval()

    # Concatenate backbone network with downstream nets
    net = Pipe(backbone_network, feature_converter=[gesture_classifier,
                                                    met_value_converter])

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
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        sense.display.DisplayTopKClassificationOutputs(top_k=1,
                                                       threshold=0.5),
        sense.display.DisplayMETandCalories(y_offset=40),
    ]
    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops,
                                                   border_size=50, display_fn=display_fn)

    # Run live inference
    controller = Controller(
        neural_network=net,
        post_processors=post_processors,
        results_display=display_results,
        callbacks=[],
        camera_id=camera_id,
        path_in=path_in,
        path_out=path_out,
        use_gpu=use_gpu,
        stop_event=stop_event,
    )
    controller.run_inference()


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    _weight = float(args['--weight'])
    _height = float(args['--height'])
    _age = float(args['--age'])
    _gender = args['--gender'] or None
    _model_name = args['--model_name'] or None
    _model_version = args['--model_version'] or None
    _use_gpu = args['--use_gpu']
    _camera_id = int(args['--camera_id'] or 0)
    _path_in = args['--path_in'] or None
    _path_out = args['--path_out'] or None
    _title = args['--title'] or None

    run_fitness_tracker(
        model_name=_model_name,
        model_version=_model_version,
        path_in=_path_in,
        path_out=_path_out,
        weight=_weight,
        height=_height,
        age=_age,
        gender=_gender,
        title=_title,
        camera_id=_camera_id,
        use_gpu=_use_gpu
    )

