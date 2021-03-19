import os
import subprocess
import shlex
import time
import urllib

from flask import Blueprint
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for
from flask_socketio import emit

from tools.sense_studio import utils
from tools.sense_studio import socketio

training_bp = Blueprint('training_bp', __name__)

train_process = None


@training_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)
    return render_template('training.html', project=project, path=path, models=utils.BACKBONE_MODELS, is_disabled=False,
                           message="Train Model...")


@training_bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.form
    project = data['project']
    path = data['path']
    num_layers_to_finetune = data['layers_to_finetune']
    output_folder = data['output_folder']
    model_name = data['model_name']
    epochs = data['epochs']
    config = utils.load_project_config(path)

    model_version = model_name.split('-')[1]
    model_name = model_name.split('-')[0]
    path_out = os.path.join(path, output_folder, 'checkpoints')
    os.makedirs(path_out, exist_ok=True)

    train_classifier = ["python tools/train_classifier.py", f"--path_in={path}",
                        f"--num_layers_to_finetune={num_layers_to_finetune}",
                        "--use_gpu" if config['use_gpu'] else "",
                        f"--path_out={path_out}" if path_out else "",
                        f"--model_name={model_name}" if model_name else "",
                        f"--model_version={model_version}" if model_version else "",
                        f"--epochs={epochs}",
                        "--overwrite"]

    train_classifier = ' '.join(train_classifier)
    train_classifier = shlex.split(train_classifier)

    global train_process
    train_process = subprocess.Popen(train_classifier, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

    return render_template('training.html', project=project, path=path, models=utils.BACKBONE_MODELS,
                           is_disabled=True, output_folder=output_folder, message="Train Model...")


@training_bp.route('/cancel-training', methods=['POST'])
def cancel_training():
    data = request.form
    project = data['project']
    path = data['path']
    output_folder = data['output_folder']
    global train_process
    if train_process:
        train_process.terminate()
        train_process = None
        message = "Training Cancelled."
    return render_template('training.html', project=project, path=path, models=utils.BACKBONE_MODELS,
                           is_disabled=False, message=message, output_folder=output_folder)


@socketio.on('training_logs', namespace='/train-model')
def send_training_logs(msg):
    global train_process
    if train_process:
        while True:
            output = train_process.stdout.readline()
            if output == b'' and train_process.poll() is not None:
                train_process.terminate()
                train_process = None
                break
            if output:
                time.sleep(0.1)
                emit('training_logs', {'log': output.decode().strip() + '\n'})
        img_path = url_for('training_bp.confusion_matrix', project=msg['project'])
        emit('success', {'status': 'Complete', 'img_path': img_path})
    else:
        emit('status', msg)


@training_bp.route('/confusion-matrix/<string:project>', methods=['GET'])
@training_bp.route('/confusion-matrix/<string:project>/<string:output_folder>', methods=['GET'])
def confusion_matrix(project, output_folder):
    path = utils.lookup_project_path(project)
    img_path = os.path.join(path, output_folder, 'checkpoints')
    return send_from_directory(img_path, filename='confusion_matrix.png', as_attachment=True)
