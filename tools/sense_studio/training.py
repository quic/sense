import multiprocessing
import os
import queue
import urllib

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for
from flask_socketio import emit

from tools.sense_studio import utils
from tools.sense_studio import socketio
from tools.train_classifier import train_model

training_bp = Blueprint('training_bp', __name__)

train_process = None
queue_train_logs = None


@training_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)
    output_path_prefix = os.path.join(os.path.basename(path), 'checkpoints', '')
    return render_template('training.html', project=project, path=path, models=utils.BACKBONE_MODELS,
                           output_path_prefix=output_path_prefix)


@training_bp.route('/start-training', methods=['POST'])
def start_training():
    data = request.json
    path = data['path']
    num_layers_to_finetune = data['layersToFinetune']
    output_folder = data['outputFolder']
    model_name = data['modelName']
    epochs = data['epochs']

    config = utils.load_project_config(path)
    model_name, model_version = model_name.split('-')
    path_out = os.path.join(path, 'checkpoints', output_folder)

    ctx = multiprocessing.get_context('spawn')

    global queue_train_logs
    queue_train_logs = ctx.Queue()

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
    }

    global train_process
    train_process = ctx.Process(target=train_model, kwargs=training_kwargs)
    train_process.start()

    return jsonify(success=True)


@training_bp.route('/cancel-training', methods=['POST'])
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

    path = utils.lookup_project_path(project)
    img_path = os.path.join(path, 'checkpoints', output_folder, 'confusion_matrix.png')
    if os.path.exists(img_path):
        img_path = url_for('training_bp.confusion_matrix',
                           project=msg['project'],
                           output_folder=msg['outputFolder'])
        emit('success', {'status': 'Complete', 'img_path': img_path})
    else:
        emit('failed', {'status': 'Failed'})


@training_bp.route('/confusion-matrix/<string:project>/', methods=['GET'])
@training_bp.route('/confusion-matrix/<string:project>/<string:output_folder>', methods=['GET'])
def confusion_matrix(project, output_folder=""):
    project = urllib.parse.unquote(project)
    output_folder = urllib.parse.unquote(output_folder)
    path = utils.lookup_project_path(project)
    img_path = os.path.join(path, 'checkpoints', output_folder)
    return send_from_directory(img_path, filename='confusion_matrix.png', as_attachment=True)
