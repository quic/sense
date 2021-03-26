import urllib

from flask import Blueprint
from flask import render_template

from tools.sense_studio import project_utils

testing_bp = Blueprint('testing_bp', __name__)

test_process = None


@testing_bp.route('/<string:project>', methods=['GET'])
def testing_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    return render_template('testing.html', project=project, path=path, output_folder="video_output")
