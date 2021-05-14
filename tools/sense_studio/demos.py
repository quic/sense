import os
import importlib
import urllib

from flask import render_template
from flask import Blueprint

from tools.sense_studio import project_utils
from tools.sense_studio import socketio
from tools.sense_studio import utils

demos_bp = Blueprint('demos_bp', __name__)


@demos_bp.route('/<string:project>', methods=['GET'])
def demos_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    project_config = project_utils.load_project_config(path)
    output_path_prefix = os.path.join(os.path.basename(path), 'checkpoints', '')
    return render_template('demos.html', project=project, path=path, models=utils.get_available_backbone_models(),
                           output_path_prefix=output_path_prefix, project_config=project_config)

