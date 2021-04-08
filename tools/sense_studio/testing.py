import base64
import glob
import multiprocessing
import os
import queue
import urllib

import cv2
from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask_socketio import emit
from natsort import natsorted
from natsort import ns

from tools.sense_studio.custom_classifier_script import run_custom_classifier
from tools.sense_studio import project_utils
from tools.sense_studio import socketio

testing_bp = Blueprint('testing_bp', __name__)

test_process = None
queue_testing_output = None


@testing_bp.route('/<string:project>', methods=['GET'])
def testing_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    output_path_prefix = os.path.join(os.path.basename(path), 'output_videos', '')

    default_checkpoint = os.path.join(path, 'checkpoints', 'best_classifier.checkpoint')
    # If classifier checkpoint exist, get the path of sub-directory from checkpoints
    classifiers = [os.path.join('checkpoints', os.path.basename(d)) for d in glob.glob(f"{path}/checkpoints/*")
                   if os.path.exists(os.path.join(d, 'best_classifier.checkpoint'))
                   and os.path.isdir(d)]
    classifiers.extend(['checkpoints/'] if os.path.exists(default_checkpoint) else [])
    classifiers = natsorted(classifiers, alg=ns.IC)

    return render_template('testing.html', project=project, path=path, output_path_prefix=output_path_prefix,
                           classifiers=classifiers)


@testing_bp.route('/start-testing', methods=['POST'])
def start_testing():
    data = request.json
    path_in = data['inputVideoPath'] or ''
    path_out = data['outputVideoName'] or ''
    title = data['title']
    path = data['path']
    custom_classifier = os.path.join(path, data['classifier'])

    config = project_utils.load_project_config(path)
    output_dir = os.path.join(path, 'output_videos')
    os.makedirs(output_dir, exist_ok=True)

    if path_out:
        path_out = os.path.join(output_dir, path_out + '.mp4')

    ctx = multiprocessing.get_context('spawn')

    global queue_testing_output
    queue_testing_output = ctx.Queue()

    testing_kwargs = {
        'path_in': path_in,
        'path_out': path_out,
        'custom_classifier': custom_classifier,
        'title': title,
        'use_gpu': config['use_gpu'],
        'video_frames': queue_testing_output.put,
    }

    global test_process
    test_process = ctx.Process(target=run_custom_classifier, kwargs=testing_kwargs)
    test_process.start()

    return jsonify(success=True)


@testing_bp.route('/cancel-testing')
def cancel_testing():
    global test_process
    if test_process:
        test_process.terminate()
        test_process = None

    return jsonify(success=True)


@socketio.on('stream_video', namespace='/stream-video')
def stream_video(msg):
    global test_process
    global queue_testing_output
    try:
        while test_process.is_alive():
            try:
                output = queue_testing_output.get(timeout=1)
                if type(output) == str:
                    emit('testing_logs', {'log': output})
                else:
                    # Encode frame as jpeg
                    frame = cv2.imencode('.jpg', output)[1].tobytes()
                    # Encode frame in base64 version and remove utf-8 encoding
                    frame = base64.b64encode(frame).decode('utf-8')
                    emit('testing_images', {'image': f"data:image/jpeg;base64,{frame}"})
            except queue.Empty:
                # No message received during the last second
                pass

        test_process.terminate()
        test_process = None
    except AttributeError:
        # test_process has been cancelled and is None
        pass
    finally:
        queue_testing_output.close()

    emit('success', {'status': 'Complete', 'image': ""})
