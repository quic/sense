#!/usr/bin/env python
"""
Run a model that can classify and count different volleyball techniques.

Usage:
  run_volleyball_counter.py [--camera_id=CAMERA_ID]
                            [--path_in=FILENAME]
                            [--path_out=FILENAME]
                            [--title=TITLE]
                            [--use_gpu]
  run_volleyball_counter.py (-h | --help)

Options:
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
  --use_gpu                  Whether to run inference on the GPU or not.
"""
from docopt import docopt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks.nn_utils import Pipe, LogisticRegression
from sense.downstream_tasks.postprocess import AggregatedPostProcessors
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.downstream_tasks.postprocess import TwoPositionsCounter
from sense.downstream_tasks.volleyball import INT2LAB, LAB2INT
from sense.loading import build_backbone_network
from sense.loading import get_relevant_weights
from sense.loading import ModelConfig


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['volleyball_counter']),
]


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    camera_id = int(args['--camera_id'] or 0)
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    title = args['--title'] or None
    use_gpu = args['--use_gpu']

    # Load weights
    selected_config, weights = get_relevant_weights(SUPPORTED_MODEL_CONFIGURATIONS)

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'])
    # TODO: Update with fine-tuned weights from checkpoint

    counter = LogisticRegression(num_in=backbone_network.feature_dim,
                                 num_out=len(INT2LAB))
    counter.load_state_dict(weights['volleyball_counter'], strict=False)
    counter.eval()

    # Concatenate backbone network and volleyball counter
    net = Pipe(backbone_network, counter)

    postprocessor = [
        # v1: 0.1
        # OnePositionRepCounter(mapping=class2int,
        #                       position='forearm_passing_position_1',
        #                       threshold=0.05,
        #                       out_key='forearm passes'),
        # OnePositionRepCounter(mapping=class2int,
        #                       position='overhead_passing_position_1',
        #                       threshold=0.1,
        #                       out_key='overhead passes'),
        # OnePositionRepCounter(mapping=class2int,
        #                       position='pokey_position_1',
        #                       threshold=0.2,
        #                       out_key='pokeys'),
        # OnePositionRepCounter(mapping=class2int,
        #                       position='one_arm_passing_position_1',
        #                       threshold=0.1,
        #                       out_key='one arm passes'),
        # OnePositionRepCounter(mapping=class2int,
        #                       position='bouncing_ball_position_1',
        #                       threshold=0.2,
        #                       out_key='bounces'),
        # OnePositionRepCounter(mapping=class2int,
        #                       position='dropping_ball_position_1',
        #                       threshold=0.1,
        #                       out_key='dropped'),
        # v1: 0.2
        AggregatedPostProcessors(post_processors=[
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
        PostprocessClassificationOutput(INT2LAB, smoothing=4),
    ]

    display_ops = [
        sense.display.DisplayTopKClassificationOutputs(top_k=3, threshold=0),
        sense.display.DisplayCounts(y_offset=120),
    ]
    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops, border_size=100)

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
