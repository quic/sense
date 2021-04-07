#!/usr/bin/env python
"""
Real time detection of 30 hand gestures.

Usage:
  run_gesture_recognition.py [--camera_id=CAMERA_ID]
                             [--path_in=FILENAME]
                             [--path_out=FILENAME]
                             [--title=TITLE]
                             [--model_name=NAME]
                             [--model_version=VERSION]
                             [--use_gpu]
                             [--class_names=CLASS_NAMES]
                             [--choose]
                             [--choose_all]
  run_gesture_recognition.py (-h | --help)

Options:
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
  --model_name=NAME          Name of the model to be used.
  --model_version=VERSION    Version of the model to be used.
  --use_gpu                  Whether to run inference on the GPU or not.
  --class_names=CLASS_NAMES  Specific class-names to visualise/test
  --choose                   Choose classes from a list
  --choose_all              Select all classes for testing
"""
from docopt import docopt
from PyInquirer import prompt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks.gesture_recognition import INT2LAB_reactive
from sense.downstream_tasks.gesture_recognition import INT2LAB_reactive9
from sense.downstream_tasks.gesture_recognition import LAB2INT_reactive
from sense.downstream_tasks.gesture_recognition import LAB2INT_reactive9
from sense.downstream_tasks.gesture_recognition import INT2LAB_LOCAL
from sense.downstream_tasks.gesture_recognition import LAB2INT_LOCAL
from sense.downstream_tasks.gesture_recognition import LAB_THRESHOLDS_reactive
from sense.downstream_tasks.gesture_recognition import LAB_THRESHOLDS_reactive9
from sense.downstream_tasks.nn_utils import LogisticRegression, MultiTimestepsLogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.downstream_tasks.postprocess import OnePositionRepCounter
from sense.loading import get_relevant_weights
from sense.loading import build_backbone_network
from sense.loading import ModelConfig


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['gesture_reactive']),
    ModelConfig('StridedInflatedEfficientNet', 'reactive_gesture_demo', ['gesture_reactive']),
    ModelConfig('StridedInflatedEfficientNet', 'reactive_gesture_demo_fps', ['gesture_reactive']),
    ModelConfig('StridedInflatedEfficientNet', 'reactive_gesture_demo_fps_v2', ['gesture_reactive']),
    ModelConfig('StridedInflatedEfficientNet', 'reactive_gesture_demo_fps_v3', ['gesture_reactive']),
    ModelConfig('StridedInflatedMobileNetV2', 'pro', ['gesture_recognition']),
    ModelConfig('StridedInflatedEfficientNet', 'lite', ['gesture_recognition']),
    ModelConfig('StridedInflatedMobileNetV2', 'lite', ['gesture_recognition']),
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
    class_names = args['--class_names'] or 'thumb_up,thumb_down'
    choose = args['--choose']
    choose_all = args['--choose_all']

    answer_model_select = prompt({
        'type': 'list',
        'name': 'model_select',
        'message': 'Select the model to use for testing',
        'choices': [
            'Baseline',
            'Fine-tuned',
            'FPS-variation',
            'Latest'
        ],
    })

    model_select = answer_model_select['model_select']

    LAB2INT = LAB2INT_reactive
    INT2LAB = INT2LAB_reactive
    LAB_THRESHOLDS = LAB_THRESHOLDS_reactive

    dict_name = 'gesture_reactive'
    if model_select != "Baseline":
        LAB2INT = LAB2INT_reactive9
        INT2LAB = INT2LAB_reactive9
        LAB_THRESHOLDS = LAB_THRESHOLDS_reactive9
        model_version = 'reactive_gesture_demo'
        if model_select == "FPS-variation":
            model_version = 'reactive_gesture_demo_fps'

        elif model_select == "Latest":
            model_version = 'reactive_gesture_demo_fps_v3'
            LAB2INT = LAB2INT_LOCAL
            INT2LAB = INT2LAB_LOCAL
            LAB_THRESHOLDS = {key: 0.5 for key in LAB2INT_LOCAL}

    # Load weights
    selected_config, weights = get_relevant_weights(
        SUPPORTED_MODEL_CONFIGURATIONS,
        model_name,
        model_version
    )

    classes = []
    if choose:
        answer_classes = prompt({
            'type': 'checkbox',
            'name': 'classes_chosen',
            'message': 'Choose the classes to test:',
            'choices': [{'name': key} for key in LAB2INT.keys()],
            #'choices': [{'name': key.split('=')[0]} for key in LAB2INT.keys() if key.endswith("=end")],
        })

        classes = [f"{key}" for key in answer_classes['classes_chosen']]

    if not choose and not choose_all or len(classes) == 0:
        _classes = class_names.split(',')
        prefix = 'counting - '
        suffix_start = '=start'
        suffix_end = '=end'

        classes = [prefix + cname + suffix_end for cname in _classes]

    if choose_all:
        classes = [key for key in LAB2INT.keys() if key.endswith('=end')]

    answer_options = prompt({
        'type': 'list',
        'name': 'options',
        'message': 'Choose extra display options',
        'choices': [
            'Rep Counts',
            'Probability Bar',
            'None'
        ],
    })

    options = answer_options['options']

    print('=' * 65)
    print(f"   Classes testing: ")
    print('-' * 65)
    for cname in classes:
        print(f"\t{cname}\t\t")
    print('=' * 65)

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'])

    # Create a logistic regression classifier
    gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                            num_out=len(INT2LAB))
    # gesture_classifier = MultiTimestepsLogisticRegression(num_in=backbone_network.feature_dim,
    #                                                       num_out=len(INT2LAB),
    #                                                       kernel=5)
    gesture_classifier.load_state_dict(weights[dict_name])
    gesture_classifier.eval()

    # Concatenate backbone network and logistic regression
    net = Pipe(backbone_network, gesture_classifier)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=1),
    ]

    border_size = 200

    display_ops = [
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        # sense.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0.5),
        # sense.display.DisplayClassnameOverlay(thresholds=LAB_THRESHOLDS,
        #                                       border_size=border_size if not title else border_size + 50),
        # sense.display.DisplayGraph(thresholds=LAB_THRESHOLDS, classes=classes, base_class=classes[0]),
    ]

    if options == "Rep Counts":
        for cname in classes:
            postprocessor.append(OnePositionRepCounter(LAB2INT, cname, LAB_THRESHOLDS[cname], cname))
        display_ops.append(sense.display.DisplayRepCounts2(keys=[cname for cname in classes], y_offset=0))

    if options == "Probability Bar":
        classes.append('counting - background')
        display_ops.append(sense.display.DisplayProbBar(keys=[cname for cname in classes], thresholds=LAB_THRESHOLDS))

    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops, window_size=(720, 1280))

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
