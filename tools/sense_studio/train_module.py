import os
import subprocess
import shlex
import urllib
import time

import flask
from flask import Blueprint, url_for
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
    models = None
    if os.path.exists(utils.BACKBONE_MODELS_DIR):
        models = os.listdir(utils.BACKBONE_MODELS_DIR)
    return render_template('train.html', project=project, models=models)


def stream_template(template_name, **context):
    # Ref: https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/
    current_app.update_template_context(context)
    t = current_app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    return rv


@train_bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.form

    cmd = "python tools/train_classifier.py --path_in=dataset/SoccerSkills --num_layers_to_finetune=9 --use_gpu --overwrite"
    cmd_split = shlex.split(cmd)

    global PROCESS
    PROCESS = subprocess.Popen(cmd_split, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate():
        while True:
            output = PROCESS.stdout.readline()
            if output == b'' and PROCESS.poll() is not None:
                PROCESS.terminate()
                break
            if output:
                # print(output.decode())
                time.sleep(0.1)
                yield output.decode().strip() + '\n'

    return flask.Response(stream_with_context(stream_template('train.html', project=data['project'],
                                                              models=data['models'], logs=generate())))


@train_bp.route('/cancel-training', methods=['POST'])
def cancel_training():
    data = request.form

    global PROCESS
    PROCESS.terminate()

    return render_template('train.html', project=data['project'], logs=["Training Cancelled"])
