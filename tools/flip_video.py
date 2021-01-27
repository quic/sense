#!/usr/bin/env python
"""
This script helps to flip videos horizontally for data augmentation.

Generally, it can be used to quickly double the size of your dataset, or,
in the case where you've collected data for an action performed on a specific side,
you can flip these videos and use them to classify the opposite side.

Usage:
  flip_video.py --path_in=PATH_IN
                [--path_out=PATH_OUT]
  flip_video.py (-h | --help)

Options:
  --path_in=PATH_IN     Path to the folder containing videos to be flipped
  --path_out=PATH_OUT   Path to the folder to save flipped videos
"""

import ffmpeg
import os

from docopt import docopt
from os.path import join

if __name__ == '__main__':
    # Parse arguments
    args = docopt(__doc__)
    videos_path_in = join(os.getcwd(), args['--path_in'])
    videos_path_out = join(os.getcwd(), args['--path_out']) if args.get('--path_out') else videos_path_in
    # Training script expects videos in MP4 format
    VIDEO_EXT = '.mp4'

    # Create directory to save flipped videos
    os.makedirs(videos_path_out, exist_ok=True)

    for video in os.listdir(videos_path_in):
        print(f'Processing video: {video}')
        flipped_video_name = video.split('.')[0] + '_flipped' + VIDEO_EXT
        # Original video as input
        original_video = ffmpeg.input(join(videos_path_in, video))
        # Do horizontal flip
        flipped_video = ffmpeg.hflip(original_video)
        # Get flipped video output
        flipped_video_output = ffmpeg.output(flipped_video, filename=join(videos_path_out, flipped_video_name))
        # Run to render and save video
        ffmpeg.run(flipped_video_output)

    print("Processing complete!")
