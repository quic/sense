import io
import os
import subprocess
import shlex
import time
import urllib
from multiprocessing import Process
import sys

import flask
from flask import Blueprint
from flask import current_app
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import stream_with_context

from tools.sense_studio import utils
from tools.sense_studio.training_script import training_model

training_bp = Blueprint('training_bp', __name__)

PROCESS = None


@training_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)
    return render_template('training.html', project=project, path=path, models=utils.BACKBONE_MODELS, is_disabled=False)


# def training_model_helper(log_path, **kwargs):
#     sys.stdout = open(log_path, "w")
#     training_model(**kwargs)


@training_bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.form
    training_args = {}
    project = data['project']
    path = data['path']
    config = utils.load_project_config(path)
    training_args['path_in'] = path
    training_args['num_layers_to_finetune'] = int(data['layers_to_finetune'])
    training_args['path_out'] = os.path.join(path, data['output_folder'] or "checkpoints")
    model_name = data['model_name']
    training_args['model_version'] = model_name.split('-')[1]
    training_args['model_name'] = model_name.split('-')[0]
    training_args['epochs'] = int(data['epochs'])
    training_args['use_gpu'] = config['use_gpu']

    # TODO: Had this because was thinking to write terminal logs into file and read from it.
    log_path = os.path.join(training_args['path_out'], str(os.getpid()) + ".out")

    is_disabled = True
    global PROCESS
    PROCESS = Process(target=training_model, kwargs=training_args)
    PROCESS.start()

    def get_training_logs():
        global PROCESS
        count = 0
        # TODO: Get the stdout (terminal prints) from the running PROCESS.
        while PROCESS.is_alive():
            time.sleep(0.1)
            count += 1
            yield "Yashesh: " + str(count)
        else:
            # TODO: This is_disabled assignment is not working and still in progress.
            is_disabled = False
            PROCESS.terminate()
            PROCESS = None

    return render_template('training.html', project=project, path=path,
                            is_disabled=is_disabled,
                            models=utils.BACKBONE_MODELS,
                            logs=get_training_logs())


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
