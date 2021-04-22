import urllib

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for

from tools.sense_studio import project_utils

tags_operations_bp = Blueprint('tags_operations_bp', __name__)


@tags_operations_bp.route('/<string:project>', methods=['GET'])
def tags_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    project_config = project_utils.load_project_config(path)
    project_tags = project_config['project_tags']
    project_tags = {v: k for k, v in project_tags.items()}

    return render_template('tags_operations.html', path=path, project=project, project_tags=project_tags)


@tags_operations_bp.route('/create-project-tag', methods=['POST'])
def create_tag_in_project_tags():
    data = request.form
    project = data['project']
    path = data['path']
    project_config = project_utils.load_project_config(path)
    tag_name = data['tag']

    project_tags = project_config.get('project_tags', {})
    if project_tags:
        max_tag_index = max(project_tags.items(), key=lambda kv: kv[1])[1]
        project_tags[tag_name] = max_tag_index + 1
    else:
        project_tags[tag_name] = 0

    project_config['project_tags'] = project_tags
    project_utils.write_project_config(path, project_config)
    return redirect(url_for('tags_operations_bp.tags_page', project=project))


@tags_operations_bp.route('/remove-project-tag', methods=['POST'])
def remove_tag_from_project_tags():
    data = request.json
    path = data['path']
    tag_idx = data['tagIdx']
    project_config = project_utils.load_project_config(path)
    project_tags = project_config['project_tags']

    project_tags = {tag_name: tag_index for tag_name, tag_index in project_tags.items() if tag_index != int(tag_idx)}
    project_config['project_tags'] = project_tags

    project_utils.write_project_config(path, project_config)
    return jsonify(success=True)


def edit_tag_in_project_tags():
    pass
