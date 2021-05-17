import base64
import importlib
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

from tools.sense_studio import project_utils
from tools.sense_studio import socketio
from tools.sense_studio import utils

examples_bp = Blueprint('examples_bp', __name__)

example_process: Optional[multiprocessing.Process] = None
queue_example_output: Optional[multiprocessing.Queue] = None
stop_event: Optional[multiprocessing.Event] = None


@examples_bp.route('/<string:project>', methods=['GET'])
def examples_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    project_config = project_utils.load_project_config(path)
    output_path_prefix = os.path.join(os.path.basename(path), 'checkpoints', '')

    examples = project_utils.get_examples()
    return render_template('examples.html', project=project, path=path, output_path_prefix=output_path_prefix,
                           project_config=project_config, models=utils.get_available_backbone_models(),
                           examples=examples)


@examples_bp.route('/start-running-example', methods=['POST'])
def start_running_example():
    data = request.json
    path_in = data['inputVideoPath']
    output_video_name = data['outputVideoName']
    title = data['title']
    path = data['path']
    model_name = data['modelName']
    model_name, model_version = model_name.split('-')

    config = project_utils.load_project_config(path)

    if output_video_name:
        output_dir = os.path.join(path, 'output_videos')
        os.makedirs(output_dir, exist_ok=True)
        path_out = os.path.join(output_dir, output_video_name + '.mp4')
    else:
        path_out = None

    ctx = multiprocessing.get_context('spawn')

    global queue_example_output
    global stop_event
    queue_example_output = ctx.Queue()
    stop_event = ctx.Event()

    example_kwargs = {
        'model_version': model_version,
        'model_name': model_name,
        'weight': int(data['weight']),
        'height': int(data['height']),
        'gender': data['gender'],
        'age': int(data['age']),
        'path_in': path_in,
        'path_out': path_out,
        'title': title,
        'use_gpu': config['use_gpu'],
        'display_fn': queue_example_output.put,
        'stop_event': stop_event,
    }

    examples = project_utils.get_examples()

    example_script = importlib.import_module(examples[int(data['example'])])
    print(example_script)
    print(getattr(f'examples.{example_script}', 'run_example'))
    global example_process
    example_process = ctx.Process(target=getattr(f'examples.{example_script}', 'run_example'), kwargs=example_kwargs)
    example_process.start()

    return jsonify(success=True)


@examples_bp.route('/cancel-running_example')
def cancel_running_example():
    global example_process
    global stop_event
    if example_process:
        # Send signal to stop inference
        stop_event.set()
        # Wait until process is complete
        example_process.join()
        example_process = None
        stop_event.clear()

    return jsonify(success=True)


@socketio.on('stream_video', namespace='/stream-video')
def stream_video(msg):
    global example_process
    global queue_example_output

    try:
        while example_process.is_alive():
            try:
                output = queue_example_output.get(timeout=1)
                if type(output) == str:
                    emit('testing_logs', {'log': output})
                else:
                    # Encode frame as jpeg
                    frame = cv2.imencode('.jpg', output)[1].tobytes()
                    # Encode frame in base64 version and remove utf-8 encoding
                    frame = base64.b64encode(frame).decode('utf-8')
                    emit('stream_frame', {'image': f'data:image/jpeg;base64,{frame}'})
            except queue.Empty:
                # No message received during the last second
                pass

        example_process.terminate()
        example_process = None
    except AttributeError:
        # example_process has been cancelled and is None
        pass
    finally:
        queue_example_output.close()

    emit('success', {'status': 'Complete'})

