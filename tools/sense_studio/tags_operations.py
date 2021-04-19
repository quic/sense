import urllib

from flask import Blueprint
from flask import render_template
from flask import url_for


tags_operations_bp = Blueprint('tags_operations_bp', __name__)


@tags_operations_bp.route('/<string:project>', methods=['GET'])
def tags_page(project):
    project = urllib.parse.unquote(project)
    return render_template('tags_operations.html', project=project)
