#!/usr/bin/env python
"""
This script helps to flip videos horizontally for data augmentation. 

Generally, it can be used to quickly double the size of your dataset, or, in the case where you've collected data for an action performed on a specific side, you can flip these videos and use them to classify the opposite side.

Usage:
  flip_video.py --path_in=PATH_IN --path_out=PATH_OUT
  flip_video.py (-h | --help)

Options:
  --path_in=PATH_IN     Full path to the videos folder
  --path_out=PATH_OUT   Full path to save flipped videos
"""

import ffmpeg
from docopt import docopt
from pathlib import Path
import os

if __name__ == '__main__':
    args = docopt(__doc__)
    videos_path_in = Path(args['--path_in'])
    videos_path_out = Path(args['--path_out'])

    video_suffix = '.mp4'

    os.makedirs(videos_path_out, exist_ok=True)
    for directory in videos_path_in.iterdir():

        print(f'Processing videos in folder: {directory.name}')
        output_folder = videos_path_out / directory.name
        os.makedirs(output_folder, exist_ok=True)

        for video in directory.iterdir():
            print(f'Processing video: {video.name}')

            original_video_name = video.stem
            flipped_video_name = original_video_name + '_flipped' + video_suffix

            # Original video as input
            original_video = ffmpeg.input(directory / video.name)
            # Do horizontal flip
            flipped_video = ffmpeg.hflip(original_video)
            # Get flipped video output
            flipped_video_output = ffmpeg.output(flipped_video, filename=output_folder / flipped_video_name)
            # Run to render and save video
            ffmpeg.run(flipped_video_output)

    print("Processing done!")
