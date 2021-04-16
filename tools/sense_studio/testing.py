import base64
import glob
import multiprocessing
import os
import queue
import urllib

from typing import Optional

import cv2
from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask_socketio import emit
from natsort import natsorted
from natsort import ns

from tools.run_custom_classifier import run_custom_classifier
from tools.sense_studio import project_utils
from tools.sense_studio import socketio

testing_bp = Blueprint('testing_bp', __name__)

test_process: Optional[multiprocessing.Process] = None
queue_testing_output: Optional[multiprocessing.Queue] = None
signal_event: Optional[multiprocessing.Event] = None


@testing_bp.route('/<string:project>', methods=['GET'])
def testing_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    output_path_prefix = os.path.join(os.path.basename(path), 'output_videos', '')

    classifiers = []
    default_checkpoint = os.path.join(path, 'checkpoints', 'best_classifier.checkpoint')
    if os.path.exists(default_checkpoint):
        classifiers.append('checkpoints/')

    # If classifier checkpoint exist, get the path of sub-directory from checkpoints
    classifiers.extend([os.path.join('checkpoints', d) for d in os.listdir(os.path.join(path, 'checkpoints'))
                        if os.path.exists(os.path.join(path, 'checkpoints', d, 'best_classifier.checkpoint'))
                        and os.path.isdir(os.path.join(path, 'checkpoints', d))])

    classifiers = natsorted(classifiers, alg=ns.IC)

    return render_template('testing.html', project=project, path=path, output_path_prefix=output_path_prefix,
                           classifiers=classifiers)


@testing_bp.route('/start-testing', methods=['POST'])
def start_testing():
    data = request.json
    path_in = data['inputVideoPath']
    path_out = data['outputVideoName']
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
    global signal_event
    queue_testing_output = ctx.Queue()
    signal_event = ctx.Event()

    testing_kwargs = {
        'path_in': path_in,
        'path_out': path_out,
        'custom_classifier': custom_classifier,
        'title': title,
        'use_gpu': config['use_gpu'],
        'display_fn': queue_testing_output.put,
        'signal_event': signal_event,
    }

    global test_process
    test_process = ctx.Process(target=run_custom_classifier, kwargs=testing_kwargs)
    test_process.start()

    return jsonify(success=True)


@testing_bp.route('/cancel-testing')
def cancel_testing():
    global test_process
    global signal_event
    if test_process:
        # Send signal to stop inference
        signal_event.set()
        # Wait until process is complete
        test_process.join()
        test_process = None
        signal_event.clear()

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
                    emit('stream_frame', {'image': f"data:image/jpeg;base64,{frame}"})
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

    emit('success', {'status': 'Complete'})