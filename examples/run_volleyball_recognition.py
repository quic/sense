#!/usr/bin/env python
"""
Run models that can classify and count different volleyball techniques.

Usage:
  run_volleyball_recognition.py [--counter]
                                [--camera_id=CAMERA_ID]
                                [--path_in=FILENAME]
                                [--path_out=FILENAME]
                                [--use_gpu]
  run_volleyball_recognition.py (-h | --help)

Options:
  --counter                  Run a counting model instead of a classification model
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --use_gpu                  Whether to run inference on the GPU or not.
"""
from docopt import docopt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks.nn_utils import Pipe, LogisticRegression
from sense.downstream_tasks.postprocess import AggregatedPostProcessors
from sense.downstream_tasks.postprocess import EventCounter
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.downstream_tasks.postprocess import TwoPositionsCounter
from sense.downstream_tasks.volleyball import CLASSIFICATION_THRESHOLDS
from sense.downstream_tasks.volleyball import INT2LAB_CLASSIFICATION, INT2LAB_COUNTING
from sense.downstream_tasks.volleyball import LAB2INT_CLASSIFICATION, LAB2INT_COUNTING
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['volleyball_classifier']),
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['volleyball_counter']),
]


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    counter = args['--counter']
    camera_id = int(args['--camera_id'] or 0)
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    use_gpu = args['--use_gpu']

    head_name = 'volleyball_counter' if counter else 'volleyball_classifier'
    INT2LAB = INT2LAB_COUNTING if counter else INT2LAB_CLASSIFICATION
    LAB2INT = LAB2INT_COUNTING if counter else LAB2INT_CLASSIFICATION

    # Load weights
    selected_config, weights = get_relevant_weights(SUPPORTED_MODEL_CONFIGURATIONS, requested_converter=head_name)

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'],
                                              weights_finetuned=weights[head_name])

    head = LogisticRegression(num_in=backbone_network.feature_dim,
                              num_out=len(INT2LAB))
    head.load_state_dict(weights[head_name])
    head.eval()

    # Concatenate backbone network and head
    net = Pipe(backbone_network, head)

    # Build post-processors
    post_processors = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4),
    ]

    if counter:
        post_processors.extend([
            # AggregatedPostProcessors(
            #     post_processors=[
            #         EventCounter(key='forearm_passing_position_1',
            #                      key_idx=LAB2INT['forearm_passing_position_1'],
            #                      threshold=0.05,
            #                      out_key='forearm passes'),
            #         EventCounter(key='overhead_passing_position_1',
            #                      key_idx=LAB2INT['overhead_passing_position_1'],
            #                      threshold=0.1,
            #                      out_key='overhead passes'),
            #         EventCounter(key='pokey_position_1',
            #                      key_idx=LAB2INT['pokey_position_1'],
            #                      threshold=0.2,
            #                      out_key='pokeys'),
            #         EventCounter(key='one_arm_passing_position_1',
            #                      key_idx=LAB2INT['one_arm_passing_position_1'],
            #                      threshold=0.1,
            #                      out_key='one arm passes'),
            #         EventCounter(key='bouncing_ball_position_1',
            #                      key_idx=LAB2INT['bouncing_ball_position_1'],
            #                      threshold=0.2,
            #                      out_key='bounces'),
            #     ],
            #     out_key='counting',
            # ),
            AggregatedPostProcessors(
                post_processors=[
                    TwoPositionsCounter(pos0_idx=LAB2INT['forearm_passing_position_1'],
                                        pos1_idx=LAB2INT['forearm_passing_position_2'],
                                        threshold0=0.1,
                                        threshold1=0.05,
                                        out_key='Forearm Passes'),
                    TwoPositionsCounter(pos0_idx=LAB2INT['overhead_passing_position_1'],
                                        pos1_idx=LAB2INT['overhead_passing_position_2'],
                                        threshold0=0.18,  # v2: 0.2 v3: 0.15
                                        threshold1=0.18,  # v2: 0.2 v3: 0.15
                                        out_key='Overhead Passes'),
                    TwoPositionsCounter(pos0_idx=LAB2INT['pokey_position_1'],
                                        pos1_idx=LAB2INT['pokey_position_2'],
                                        threshold0=0.1,
                                        threshold1=0.1,
                                        out_key='Pokeys'),
                    TwoPositionsCounter(pos0_idx=LAB2INT['one_arm_passing_position_1'],
                                        pos1_idx=LAB2INT['one_arm_passing_position_2'],
                                        threshold0=0.2,  # v2: 0.1
                                        threshold1=0.2,  # v2: 0.1
                                        out_key='One Arm Passes'),
                    TwoPositionsCounter(pos0_idx=LAB2INT['bouncing_ball_position_1'],
                                        pos1_idx=LAB2INT['bouncing_ball_position_2'],
                                        threshold0=0.1,
                                        threshold1=0.1,
                                        out_key='Bounces'),
                ],
                out_key='counting',
            ),
        ])

    border_size_top = 55

    display_ops = [
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        sense.display.DisplayTopKClassificationOutputs(top_k=2, threshold=0),
    ]

    if counter:
        display_ops.append(sense.display.DisplayCounts(y_offset=border_size_top + 20))
    else:
        display_ops.append(sense.display.DisplayClassnameOverlay(thresholds=CLASSIFICATION_THRESHOLDS,
                                                                 border_size_top=border_size_top))

    display_results = sense.display.DisplayResults(display_ops=display_ops,
                                                   border_size_top=border_size_top)

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
