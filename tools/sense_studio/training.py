import os
import subprocess
import shlex
import time
import urllib

import flask
from flask import Blueprint
from flask import current_app
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import stream_with_context

from tools.sense_studio import utils

training_bp = Blueprint('training_bp', __name__)

PROCESS = None


@training_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)
    return render_template('training.html', project=project, path=path, models=utils.BACKBONE_MODELS, is_disabled=False)


def stream_template(template_name, **context):
    # Ref: https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/
    current_app.update_template_context(context)
    t = current_app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    return rv


@training_bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.form
    project = data['project']
    path = data['path']
    num_layers_to_finetune = data['layers_to_finetune']
    path_out = os.path.join(path, data['output_folder'])
    model_name = data['model_name']
    model_version = model_name.split('-')[1]
    model_name = model_name.split('-')[0]
    epochs = data['epochs']
    config = utils.load_project_config(path)

    is_disabled = True

    train_classifier = ["python tools/train_classifier.py", f"--path_in={project}",
                        f"--num_layers_to_finetune={num_layers_to_finetune}",
                        "--use_gpu" if config['use_gpu'] else "",
                        f"--path_out={path_out}" if path_out else "",
                        f"--model_name={model_name}" if model_name else "",
                        f"--model_version={model_version}" if model_version else "",
                        f"--epochs={epochs}",
                        "--overwrite"]

    train_classifier = ' '.join(train_classifier)
    train_classifier = shlex.split(train_classifier)

    global PROCESS
    PROCESS = subprocess.Popen(train_classifier, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

    def get_training_logs():
        while True:
            output = PROCESS.stdout.readline()
            if output == b'' and PROCESS.poll() is not None:
                PROCESS.terminate()
                break
            if output:
                time.sleep(0.1)
                yield output.decode().strip() + '\n'

    return flask.Response(stream_with_context(stream_template('training.html', project=project, path=path,
                                                              is_disabled=is_disabled,
                                                              models=utils.BACKBONE_MODELS,
                                                              logs=get_training_logs())))


@training_bp.route('/cancel-training', methods=['POST'])
def cancel_training():
    data = request.form
    project = data['project']
    path = data['path']
    global PROCESS
    if PROCESS:
        PROCESS.terminate()
        PROCESS = None
        log = "Training Cancelled."
    else:
        log = "No Training Process Running to Terminate."
    return render_template('training.html', project=project, path=path, models=utils.BACKBONE_MODELS, is_disabled=False,
                           logs=[log])


# @training_bp.route('/confusion-matrix/<string:project>/', methods=['GET'])
# def get_confusion_matrix(project):
#     img_path = os.path.join(os.getcwd(), 'dataset/', project)
#
#     if os.path.exists(os.path.join(img_path, 'confusion_matrix.png')):
#         return send_from_directory(img_path, 'confusion_matrix.png', as_attachment=True)
#     return None
