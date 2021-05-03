#!/usr/bin/env python
"""
Live estimation of burned calories.

Usage:
  run_calorie_estimation.py [--weight=WEIGHT --age=AGE --height=HEIGHT --gender=GENDER]
                            [--camera_id=CAMERA_ID]
                            [--path_in=FILENAME]
                            [--path_out=FILENAME]
                            [--title=TITLE]
                            [--model_name=NAME]
                            [--model_version=VERSION]
                            [--use_gpu]
  run_calorie_estimation.py (-h | --help)

Options:
  --weight=WEIGHT                 Weight (in kilograms). Will be used to convert predicted MET value to calories
                                  [default: 70]
  --age=AGE                       Age (in years). Will be used to convert predicted MET value to calories [default: 30]
  --height=HEIGHT                 Height (in centimeters). Will be used to convert predicted MET value to calories
                                  [default: 170]
  --gender=GENDER                 Gender ("male" or "female" or "other"). Will be used to convert predicted MET value to
                                  calories
  --camera_id=CAMERA_ID           ID of the camera to stream from
  --path_in=FILENAME              Video file to stream from
  --path_out=FILENAME             Video file to stream to
  --title=TITLE                   This adds a title to the window display
  --model_name=NAME               Name of the model to be used.
  --model_version=VERSION         Version of the model to be used.
  --use_gpu                       Whether to run inference on the GPU or not.
"""
from docopt import docopt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks import calorie_estimation
from sense.downstream_tasks.nn_utils import Pipe
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedMobileNetV2', 'pro', ['met_converter']),
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['met_converter']),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', ['met_converter']),
    ModelConfig('StridedInflatedEfficientNet', 'lite', ['met_converter']),
]


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    weight = float(args['--weight'])
    height = float(args['--height'])
    age = float(args['--age'])
    gender = args['--gender'] or None
    model_name = args['--model_name'] or None
    model_version = args['--model_version'] or None
    use_gpu = args['--use_gpu']

    camera_id = int(args['--camera_id'] or 0)
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    title = args['--title'] or None

    # Load weights
    selected_config, weights = get_relevant_weights(
        SUPPORTED_MODEL_CONFIGURATIONS,
        model_name,
        model_version
    )

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'])

    # Load MET value converter
    met_value_converter = calorie_estimation.METValueMLPConverter()
    met_value_converter.load_state_dict(weights['met_converter'])
    met_value_converter.eval()

    # Concatenate backbone network and met converter
    net = Pipe(backbone_network, met_value_converter)

    post_processors = [
        calorie_estimation.CalorieAccumulator(weight=weight,
                                              height=height,
                                              age=age,
                                              gender=gender,
                                              smoothing=12)
    ]

    display_ops = [
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        sense.display.DisplayDetailedMETandCalories(),
    ]
    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops)

    # Run live inference
    controller = Controller(
        neural_network=net,
        post_processors=post_processors,
        results_display=display_results,
        callbacks=[],
        camera_id=camera_id,
        path_in=path_in,
        path_out=path_out,
        use_gpu=use_gpu
    )
    controller.run_inference()
