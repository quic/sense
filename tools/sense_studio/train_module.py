import glob
import os
import subprocess
import urllib

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request

from tools.sense_studio import utils

train_bp = Blueprint('train_bp', __name__)


@train_bp.route('/<string:project>', methods=['GET'])
def training_page(project):
    project = urllib.parse.unquote(project)
    return render_template('train.html', project=project)


@train_bp.route('/train-model', methods=['POST'])
def train_model():
    data = request.form
    return render_template('train.html', project=data['project'], text="Model Training")
