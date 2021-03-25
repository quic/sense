import os
import subprocess
import shlex
import time
import urllib

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for
from flask_socketio import emit

from tools.sense_studio import project_utils
from tools.sense_studio import utils
from tools.sense_studio import socketio

training_bp = Blueprint('training_bp', __name__)

train_process = None


@training_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
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

    config = project_utils.load_project_config(path)
    model_name, model_version = model_name.split('-')
    path_out = os.path.join(path, 'checkpoints', output_folder)

    train_classifier = ["python tools/train_classifier.py", f"--path_in={path}",
                        f"--num_layers_to_finetune={num_layers_to_finetune}",
                        f"--path_out={path_out}",
                        f"--model_name={model_name}",
                        f"--model_version={model_version}",
                        f"--epochs={epochs}",
                        "--use_gpu" if config['use_gpu'] else "",
                        f"--temporal_training" if config['temporal'] else "",
                        "--overwrite"]

    train_classifier = ' '.join(train_classifier)
    train_classifier = shlex.split(train_classifier)

    global train_process
    train_process = subprocess.Popen(train_classifier, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

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
    errors = []
    if train_process:
        while True:
            try:
                errors = train_process.stderr.readlines()
                if errors:
                    for error in errors:
                        time.sleep(0.1)
                        emit('training_logs', {'log': error.decode() + '\n'})
                    train_process.terminate()
                    train_process = None
                    break
                else:
                    output = train_process.stdout.readline()
                    if output == b'' and train_process.poll() is not None:
                        train_process.terminate()
                        train_process = None
                        break
                    if output:
                        time.sleep(0.1)
                        emit('training_logs', {'log': output.decode() + '\n'})
            except AttributeError:
                # train_process has been cancelled and is None
                break

        if not errors:
            img_path = url_for('training_bp.confusion_matrix',
                               project=msg['project'],
                               output_folder=msg['outputFolder'])
            emit('success', {'status': 'Complete', 'img_path': img_path})
        else:
            emit('failed', {'status': 'Failed'})
    else:
        emit('status', msg)


@training_bp.route('/confusion-matrix/<string:project>/<string:output_folder>', methods=['GET'])
def confusion_matrix(project, output_folder):
    project = urllib.parse.unquote(project)
    output_folder = urllib.parse.unquote(output_folder)
    path = project_utils.lookup_project_path(project)
    img_path = os.path.join(path, 'checkpoints', output_folder)
    return send_from_directory(img_path, filename='confusion_matrix.png', as_attachment=True)
