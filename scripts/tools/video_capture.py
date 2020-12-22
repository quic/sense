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
                   [--no_audio]
  video_capture.py (-h | --help)

Options:
  --duration=DURATION                                   Recording time in seconds
  --pre_recording_duration=PRE_RECORDING_DURATION       Duration for the pre-recording period [default: 3]
  --num_videos=NUM_VIDEOS                               Number of videos to record [default: 1]
  --camera_id=CAMERA_ID                                 ID of the camera to stream from [default: 0]
  --path_out=PATH_OUT                                   Videos folder to stream to [default: output/]
  --file_name=FILE_NAME                                 Video filename, followed by {video_number}.mp4 [default: output]
  --no_audio                                            A flag to toggle audio alerts
"""

import os
import time

import cv2
from docopt import docopt
from os.path import join
import simpleaudio as sa

FONT = cv2.FONT_HERSHEY_PLAIN
_shutdown = False
COUNTDOWN_SOUND = 'countdown_sound.wav'
DONE_SOUND = 'done_sound.wav'
EXIT_SOUND = 'exit_sound.wav'


def _play_audio(audio_file, no_audio_alerts=False):
    """
    Plays an audio for `countdown` and `done` timer on video prompt.
    Pre-recording: count down the last three seconds, then DONE sound
    Recording: only DONE sound

    :param audio_file:          str
        Name of the audio file to play.
    :param no_audio_alerts:     boolean
        A flag to toggle audio-alerts on video-prompts
    """
    if not no_audio_alerts:
        audio_path = join(os.getcwd(), 'docs', 'audio', audio_file)     # hard-coded path to file, can be changed
        wave_obj = sa.WaveObject.from_wave_file(audio_path)
        play_obj = wave_obj.play()
        play_obj.stop()


def _capture_video(video_duration=0., record=False):
    """
    Helper method to create and show window with timer and message for recording videos, and automatically
    saving them to the desired folder with the desired file-name.

    :param video_duration:  (float)
        Time duration for the pre-recording or recording phase prompt and timer
    :param record:          (bool)
        Flag to distinguish between pre-recording and recording phases
    """
    global _shutdown
    if cap is not None:
        skip = False
        t = time.time()
        frames = []
        frame_size = (640, 480)     # default frame size
        countdown = 1 if record else min(3, int(video_duration))        # number of seconds to countdown
        margin = 0.1       # time margin (in seconds)
        time_left = video_duration - time.time() + t

        while time_left > 0:
            if not record and time_left > 0.5 and abs(countdown - time_left) <= margin:
                _play_audio(COUNTDOWN_SOUND, no_audio)
                countdown -= 1

            ret, frame = cap.read()
            frames.append(frame.copy())
            frame = cv2.flip(frame, 1)      # horizontal flip for video-preview
            frame_size = (frame.shape[1], frame.shape[0])

            if record:
                message = f"Recording video {str(index + 1)}"
            else:
                message = f"Get into position {str(index + 1)}"

            # Recording prompt
            cv2.putText(frame, message, (100, 100), FONT, 3, (255, 255, 255),
                        4, cv2.LINE_AA)
            # Recording timer
            cv2.putText(frame, f" {str(int(time_left) + 1)}",
                        (200, 250), FONT, 10, (255, 255, 255),
                        6, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            # Get key-press to skip current video or terminate script
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):      # Press `S` to skip the current video prompt
                if record:
                    print(f'\t[PROMPT]\tSkipping video {index + 1} of {num_videos}.')
                cv2.destroyAllWindows()
                skip = True
                break
            elif key == 27:                 # Press `ESC` to exit the script
                print('\t[PROMPT]\tShutting down video-recording and releasing resources.')
                _play_audio(EXIT_SOUND, no_audio)
                cv2.destroyAllWindows()
                _shutdown = True
                break

            time_left = video_duration - time.time() + t

        if not _shutdown:
            _play_audio(DONE_SOUND, no_audio)

        calculated_fps = round(len(frames) / video_duration)
        fps = 16 if calculated_fps <= 16 else calculated_fps

        if record and not skip and not _shutdown:
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
    no_audio = args['--no_audio']

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
        if _shutdown:
            break
        else:
            _capture_video(video_duration=pre_recording_duration)

        # Show timer window for actual recording
        if _shutdown:
            break
        else:
            _capture_video(video_duration=duration, record=True)

    print('Done!')
