import glob
import os
import subprocess
import urllib

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request

from tools.sense_studio import project_utils


video_recording_bp = Blueprint('video_recording_bp', __name__)


@video_recording_bp.route('/ffmpeg-check')
def check_ffmpeg():
    ffmpeg_installed = os.popen("ffmpeg -version").read()
    return jsonify(ffmpeg_installed=ffmpeg_installed != '')


@video_recording_bp.route('/record-video/<string:project>/<string:split>/<string:label>')
def record_video(project, split, label):
    """
    Display the video recording screen.
    """
    project = urllib.parse.unquote(project)
    split = urllib.parse.unquote(split)
    label = urllib.parse.unquote(label)
    path = project_utils.lookup_project_path(project)

    countdown, recording = project_utils.get_timer_default(path)

    return render_template('video_recording.html', project=project, split=split, label=label, path=path,
                           countdown=countdown, recording=recording)


@video_recording_bp.route('/save-video/<string:project>/<string:split>/<string:label>', methods=['POST'])
def save_video(project, split, label):
    project = urllib.parse.unquote(project)
    split = urllib.parse.unquote(split)
    label = urllib.parse.unquote(label)
    path = project_utils.lookup_project_path(project)

    # Read given video to a file
    input_stream = request.files['video']
    output_path = os.path.join(path, f'videos_{split}', label)
    temp_file_name = os.path.join(output_path, 'temp_video.webm')
    with open(temp_file_name, 'wb') as temp_file:
        temp_file.write(input_stream.read())

    # Find a video name that is not used yet
    existing_files = set(glob.glob(os.path.join(output_path, 'video_[0-9]*.mp4')))
    video_idx = 0
    output_file = os.path.join(output_path, f'video_{video_idx}.mp4')
    while output_file in existing_files:
        video_idx += 1
        output_file = os.path.join(output_path, f'video_{video_idx}.mp4')

    # Convert video to target frame rate and save to output name
    subprocess.call(f'ffmpeg -i "{temp_file_name}" -r 30 "{output_file}"', shell=True)

    # Remove temp video file
    os.remove(temp_file_name)

    return jsonify(success=True)
