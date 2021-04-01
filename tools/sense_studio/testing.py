import multiprocessing
import os
import queue
import urllib

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for
from flask_socketio import emit

from tools.sense_studio.custom_classifier_script import run_custom_classifier
from tools.sense_studio import project_utils
from tools.sense_studio import socketio

testing_bp = Blueprint('testing_bp', __name__)

test_process = None
queue_frames = None


@testing_bp.route('/<string:project>', methods=['GET'])
def testing_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    return render_template('testing.html', project=project, path=path, output_folder="video_output")


@testing_bp.route('/start-testing', methods=['POST'])
def start_testing():
    data = request.json
    custom_classifier = data['classifier']
    path_in = data['inputVideoPath'] or ''
    path_out = data['outputVideoName'] or ''
    title = data['title']
    path = data['path']
    output_folder = data['outputFolder']

    config = project_utils.load_project_config(path)
    path_out = os.path.join(path, output_folder, path_out)
    os.makedirs(path_out, exist_ok=True)

    ctx = multiprocessing.get_context('spawn')

    global queue_frames
    queue_frames = ctx.Queue()

    testing_kwargs = {
        'path_in': path_in,
        'path_out': path_out,
        'custom_classifier': custom_classifier,
        'title': title,
        'use_gpu': config['use_gpu'],
        # 'catch_frames': queue_frames,
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
    global queue_frames

    try:
        while test_process.is_alive():
            try:
                output = queue_frames.get(timeout=1)
                emit('training_logs', {'log': output})
            except queue.Empty:
                # No message received during the last second
                pass

        test_process.terminate()
        train_process = None
    except AttributeError:
        # train_process has been cancelled and is None
        pass
    finally:
        queue_frames.close()

    project = msg['project']
    output_folder = msg['outputFolder']

    path = project_utils.lookup_project_path(project)
    img_path = os.path.join(path, 'checkpoints', output_folder, 'confusion_matrix.png')
    if os.path.exists(img_path):
        img_path = url_for('training_bp.confusion_matrix',
                           project=msg['project'],
                           output_folder=msg['outputFolder'])
        emit('success', {'status': 'Complete', 'img_path': img_path})
    else:
        emit('failed', {'status': 'Failed'})