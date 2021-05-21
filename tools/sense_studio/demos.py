import base64
import importlib
import multiprocessing
import os
import queue

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

demos_bp = Blueprint('demos_bp', __name__)

demo_process: Optional[multiprocessing.Process] = None
queue_demo_output: Optional[multiprocessing.Queue] = None
stop_event: Optional[multiprocessing.Event] = None


@demos_bp.route('/', methods=['GET'])
def demos_page():
    output_path_prefix = os.path.join(os.path.basename(os.getcwd()), 'demo_output_videos', '')
    demos = project_utils.get_demos()
    return render_template('demos.html', output_path_prefix=output_path_prefix,
                           models=utils.get_available_backbone_models(), demos=demos)


@demos_bp.route('/start-demo', methods=['POST'])
def start_demo():
    data = request.json
    demo = data['demo']
    path_in = data['inputVideoPath']
    output_video_name = data['outputVideoName']
    title = data['title']
    model_name = data['modelName']
    model_name, model_version = model_name.split('-')

    if output_video_name:
        output_dir = os.path.join(os.getcwd(), 'demo_output_videos')
        os.makedirs(output_dir, exist_ok=True)
        path_out = os.path.join(output_dir, output_video_name + '.mp4')
    else:
        path_out = None

    ctx = multiprocessing.get_context('spawn')

    global queue_demo_output
    global stop_event
    queue_demo_output = ctx.Queue()
    stop_event = ctx.Event()

    example_kwargs = {
        'model_version': model_version,
        'model_name': model_name,
        'path_in': path_in,
        'path_out': path_out,
        'weight': float(data['weight']),
        'height': float(data['height']),
        'age': float(data['age']),
        'gender': data['gender'],
        'title': title,
        'use_gpu': data['gpuInput'],
        'display_fn': queue_demo_output.put,
        'stop_event': stop_event,
    }

    # Dynamically import the script based on demo name
    import_demo = importlib.import_module(f'examples.{demo}')
    global demo_process
    demo_process = ctx.Process(target=getattr(import_demo, demo), kwargs=example_kwargs)
    demo_process.start()

    return jsonify(success=True)


@demos_bp.route('/cancel-demo')
def cancel_demo():
    global demo_process
    global stop_event
    if demo_process:
        # Send signal to stop inference
        stop_event.set()
        # Wait until process is complete
        demo_process.join()
        demo_process = None
        stop_event.clear()

    return jsonify(success=True)


@socketio.on('stream_demo', namespace='/stream-demo')
def stream_demo(msg):
    global demo_process
    global queue_demo_output

    try:
        while demo_process.is_alive():
            try:
                output = queue_demo_output.get(timeout=1)
                if type(output) == str:
                    emit('demo_logs', {'log': output})
                else:
                    # Encode frame as jpeg
                    height, width = output.shape[:2]
                    frame = cv2.imencode('.jpg', output)[1].tobytes()
                    # Encode frame in base64 version and remove utf-8 encoding
                    frame = base64.b64encode(frame).decode('utf-8')
                    emit('stream_frame', {'image': f'data:image/jpeg;base64,{frame}', 'height': height, 'width': width})
            except queue.Empty:
                # No message received during the last second
                pass

        demo_process.terminate()
        demo_process = None
    except AttributeError:
        # demo_process has been cancelled and is None
        pass
    finally:
        queue_demo_output.close()

    emit('success', {'status': 'Complete'})


@demos_bp.route('/get-supported-models', methods=['POST'])
def get_supported_models():
    data = request.json
    demo_name = data['demo']
    import_demo = importlib.import_module(f'examples.{demo_name}')
    models = utils.get_available_backbone_models(getattr(import_demo, 'SUPPORTED_MODEL_CONFIGURATIONS'))
    return jsonify({'models': models})
