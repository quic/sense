#!/usr/bin/env python
"""
Live estimation of burned calories.
Usage:
  video_capture.py --duration=DURATION
               [--pre_recording_duration=PRE_RECORDING_DURATION]
               [--number_videos=NUMBER_VIDEOS]
               [--camera_id=CAMERA_ID]
               [--path_out=PATHOUT]
               [--file_name=FILENAME]
  video_capture.py (-h | --help)

Options:
  --duration=DURATION                                     Recording time in seconds
  --pre_recording_duration=PRE_RECORDING_DURATION        Duration for pre recording period
  --number_videos =NUMBER_VIDEOS                         Number videos to record [default: 1]
  --camera_id=CAMERA_ID                                         ID of the camera to stream from
  --path_out=PATHOUT                                Video file to stream to [default: output/]
  --file_name=FILENAME                              file name of the video. will be followed by {video number}.mp4 [default: output_]
"""

import numpy as np
import os
import cv2
import time
from docopt import docopt

if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    duration = float(args['--duration'])
    pre_recording_duration = float(args['--pre_recording_duration'])
    number_videos = int(args['--number_videos'])
    camera_id = args['--camera_id'] or 0
    path_out = args['--path_out']
    filename = args['--file_name']


fps = 30.

cap = cv2.VideoCapture(0)
os.makedirs(path_out, exist_ok=True)
for i in range(number_videos):
    file = f"{filename}_{str(i)}.mp4"
    out = None
    t = time.time()
    # TOManik: refactor this loop because same code as next one
    while True:
        ret, frame = cap.read()
        if out is None:
            out = cv2.VideoWriter(os.path.join(path_out, file), 0x7634706d, fps, (frame.shape[1],
                                                                                  frame.shape[0]))
        # TOManik: try to see if we can avoid recording here
        out.write(frame)
        FONT = cv2.FONT_HERSHEY_PLAIN

        cv2.putText(frame, f"get into position {str(i + 1)}", (100, 100), FONT, 3, (255, 255, 255),
                    2, cv2.LINE_AA)

        cv2.putText(frame, f" {str(int(pre_recording_duration - time.time() + t ))}",
                    (200, 200), FONT, 10, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - t > pre_recording_duration:
            break

    out.release()
    cv2.destroyAllWindows()
    out = None


    t = time.time()
    while True:
        ret, frame = cap.read()
        if out is None:
            # TOManik: instead of fps, try to see if we can read the camera framerate: fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(os.path.join(path_out, file), 0x7634706d, fps, (frame.shape[1],
                                                                                  frame.shape[0]))
        out.write(frame)
        FONT = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, f"recording video {str(i + 1)}", (100, 100), FONT, 3, (255, 255, 255),
                    2, cv2.LINE_AA)

        cv2.putText(frame, f" {str(int(duration - time.time() + t ))}",
                    (200, 200), FONT, 10, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - t > duration:
            break

    out.release()
    cv2.destroyAllWindows()
