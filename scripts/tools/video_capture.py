#!/usr/bin/env python
"""
This script can be used to record videos (using a connected camera source) and save them on the system.

Usage:
  video_capture.py --duration=DURATION
                   [--pre_recording_duration=PRE_RECORDING_DURATION]
                   [--num_videos=NUM_VIDEOS]
                   [--camera_id=CAMERA_ID]
                   [--path_out=PATH_OUT]
                   [--file_name=FILE_NAME]
  video_capture.py (-h | --help)

Options:
  --duration=DURATION                                   Recording time in seconds
  --pre_recording_duration=PRE_RECORDING_DURATION       Duration for the pre-recording period [default: 3]
  --num_videos=NUM_VIDEOS                               Number of videos to record [default: 1]
  --camera_id=CAMERA_ID                                 ID of the camera to stream from [default: 0]
  --path_out=PATH_OUT                                   Videos folder to stream to [default: output/]
  --file_name=FILE_NAME                                 Video filename, followed by {video_number}.mp4 [default: output]
"""

import os
import sys
import time

import cv2
from docopt import docopt

FONT = cv2.FONT_HERSHEY_PLAIN
TERMINATE = False


def _capture_video(video_duration=0., record=False):
    """
    Helper method to create and show window with timer and message for recording videos, and automatically
    saving them to the desired folder with the desired file-name.

    :param video_duration:  (float)
        Time duration for the pre-recording or recording phase prompt and timer
    :param record:          (bool)
        Flag to distinguish between pre-recording and recording phases
    """
    global TERMINATE
    if cap is not None:
        skip = False
        t = time.time()
        frames = []
        frame_size = (640, 480)     # default frame size
        while time.time() - t < video_duration:
            ret, frame = cap.read()
            frames.append(frame.copy())
            frame_size = (frame.shape[1], frame.shape[0])

            if record:
                message = f"recording video {str(index + 1)}"
            else:
                message = f"get into position {str(index + 1)}"

            # Recording prompt
            cv2.putText(frame, message, (100, 100), FONT, 3, (255, 255, 255),
                        2, cv2.LINE_AA)
            # Recording timer
            cv2.putText(frame, f" {str(int(video_duration - time.time() + t))}",
                        (200, 200), FONT, 10, (255, 255, 255),
                        2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            # Get key-press to skip current video or terminate script
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):      # Pressing `S` will skip the current video prompt
                cv2.destroyAllWindows()
                skip = True
                break
            elif key == 27:                 # `ESC` to exit the script
                cv2.destroyAllWindows()
                TERMINATE = True
                break

        calculated_fps = round(len(frames) / video_duration)
        fps = 16 if calculated_fps <= 16 else 30

        if record and not skip and not TERMINATE:
            out = cv2.VideoWriter(os.path.join(path_out, file), 0x7634706d, fps, frame_size)
            for frame in frames:
                out.write(frame)
            out.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    duration = float(args['--duration'])
    pre_recording_duration = float(args['--pre_recording_duration'])
    num_videos = int(args['--num_videos'])
    camera_id = int(args['--camera_id'])
    path_out = args['--path_out']
    filename = args['--file_name']

    cap = cv2.VideoCapture(camera_id)
    os.makedirs(path_out, exist_ok=True)
    for i in range(num_videos):
        # Video-index to be displayed in the prompt since it gets overwritten in the next steps
        index = i
        file = f"{filename}_{str(i)}.mp4"

        # Avoid overwriting pre-existing files
        while file in os.listdir(path_out):
            i += 1
            file = f"{filename}_{str(i)}.mp4"

        # Show timer window before recording
        if not TERMINATE:
            _capture_video(video_duration=pre_recording_duration)

        # Show timer window for actual recording
        if not TERMINATE:
            _capture_video(video_duration=duration, record=True)
