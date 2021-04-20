import base64
import multiprocessing
import os
import queue
import urllib

from typing import Optional

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask_socketio import emit

from tools.sense_studio import project_utils
from tools.sense_studio import utils
from tools.sense_studio import socketio
from tools.train_classifier import train_model

training_bp = Blueprint('training_bp', __name__)

train_process: Optional[multiprocessing.Process] = None
queue_train_logs: Optional[multiprocessing.Queue] = None
confmat_event: Optional[multiprocessing.Event] = None


@training_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    project_config = project_utils.load_project_config(path)
    output_path_prefix = os.path.join(os.path.basename(path), 'checkpoints', '')
    return render_template('training.html', project=project, path=path, models=utils.get_available_backbone_models(),
                           output_path_prefix=output_path_prefix, project_config=project_config)


@training_bp.route('/start-training', methods=['POST'])
def start_training():
    data = request.json
    path = data['path']
    num_layers_to_finetune = data['layersToFinetune']
    output_folder = data['outputFolder']
    model_name = data['modelName']
    epochs = data['epochs']

    config = project_utils.load_project_config(path)
    model_name, model_version = model_name.split('-')
    path_out = os.path.join(path, 'checkpoints', output_folder)

    ctx = multiprocessing.get_context('spawn')

    global queue_train_logs
    global confmat_event

    queue_train_logs = ctx.Queue()
    confmat_event = ctx.Event()

    training_kwargs = {
        'path_in': path,
        'num_layers_to_finetune': int(num_layers_to_finetune),
        'path_out': path_out,
        'model_version': model_version,
        'model_name': model_name,
        'epochs': int(epochs),
        'use_gpu': config['use_gpu'],
        'temporal_training': config['temporal'],
        'log_fn': queue_train_logs.put,
        'confmat_event': confmat_event,
    }

    global train_process
    train_process = ctx.Process(target=train_model, kwargs=training_kwargs)
    train_process.start()

    return jsonify(success=True)


@training_bp.route('/cancel-training')
def cancel_training():
    global train_process
    if train_process:
        train_process.terminate()
        train_process = None

    return jsonify(success=True)


@socketio.on('connect_training_logs', namespace='/connect-training-logs')
def send_training_logs(msg):
    global train_process
    global queue_train_logs
    global confmat_event

    try:
        while train_process.is_alive():
            try:
                output = queue_train_logs.get(timeout=1)
                emit('training_logs', {'log': output})
            except queue.Empty:
                # No message received during the last second
                pass

        train_process.terminate()
        train_process = None
    except AttributeError:
        # train_process has been cancelled and is None
        pass
    finally:
        queue_train_logs.close()

    project = msg['project']
    output_folder = msg['outputFolder']

    path = project_utils.lookup_project_path(project)
    img_path = os.path.join(path, 'checkpoints', output_folder, 'confusion_matrix.png')
    if confmat_event.is_set() and os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            data = f.read()
        img_base64 = base64.b64encode(data).decode('utf-8')
        if img_base64:
            emit('success', {'status': 'Complete', 'img': f'data:image/png;base64,{img_base64}'})
        else:
            emit('failed', {'status': 'Failed'})
    else:
        emit('failed', {'status': 'Failed'})
