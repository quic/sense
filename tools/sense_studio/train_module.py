import os
import subprocess
import shlex
import urllib
import time

import flask
from flask import Blueprint
from flask import render_template
from flask import request
from flask import current_app
from flask import stream_with_context

from tools.sense_studio import utils

train_bp = Blueprint('train_bp', __name__)

PROCESS = None


@train_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    return render_template('train.html', project=project, models=utils.BACKBONE_MODELS)


def stream_template(template_name, **context):
    # Ref: https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/
    current_app.update_template_context(context)
    t = current_app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    return rv


@train_bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.form
    project = data['project']
    num_layers_to_finetune = data['layers_to_finetune']
    path_out = data['output_folder']
    model_name = data['model_name']
    path = utils.lookup_project_path(project)
    config = utils.load_project_config(path)

    train_classifier = ["python tools/train_classifier.py", f"--path_in=/home/twentybn/Code/sense/dataset/{project}",
                        f"--num_layers_to_finetune={num_layers_to_finetune}",
                        "--use_gpu" if config['use_gpu'] else "",
                        f"--path_out={path_out}" if path_out else "",
                        f"--model_name={model_name}" if model_name else "",
                        "--overwrite"]

    train_classifier = ' '.join(train_classifier)
    train_classifier = shlex.split(train_classifier)

    global PROCESS
    PROCESS = subprocess.Popen(train_classifier, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate():
        while True:
            output = PROCESS.stdout.readline()
            if output == b'' and PROCESS.poll() is not None:
                PROCESS.terminate()
                break
            if output:
                time.sleep(0.1)
                yield output.decode().strip() + '\n'

    return flask.Response(stream_with_context(stream_template('train.html', project=data['project'],
                                                              models=utils.BACKBONE_MODELS, logs=generate())))


@train_bp.route('/cancel-training', methods=['POST'])
def cancel_training():
    data = request.form

    global PROCESS
    if PROCESS:
        PROCESS.terminate()
        log = "Training Cancelled."
    else:
        log = "No Training Process Running to Terminate."
    return render_template('train.html', project=data['project'], models=utils.BACKBONE_MODELS, logs=[log])
