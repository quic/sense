import glob
import os
import subprocess
import shlex
import urllib

import flask
from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request

from tools.sense_studio import utils

train_bp = Blueprint('train_bp', __name__)


@train_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    models = None
    if os.path.exists(utils.BACKBONE_MODELS_DIR):
        models = os.listdir(utils.BACKBONE_MODELS_DIR)
    return render_template('train.html', project=project, models=models)


@train_bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.form
    # print(data)
    return flask.Response(get_terminal_logs(), mimetype='text/html')
    # return render_template('train.html', project=data['project'], text="Model Training")
    # return "Done"


def get_terminal_logs():
    cmd = "PYTHONPATH=./ python tools/train_classifier.py --path_in=dataset/SoccerSkills --num_layers_to_finetune=9"
    cmd_split = shlex.split(cmd)
    process = subprocess.Popen(cmd_split, stdout=subprocess.PIPE, shell=True)
    print("here")

    for line in iter(process.stdout.readline, ''):
        if line:
            print(line)
            yield line.decode()

    # print(process.stdout.readlines())
    # rc = process.poll()