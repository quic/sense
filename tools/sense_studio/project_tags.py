import urllib

from flask import Blueprint
from flask import jsonify
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for

from tools.sense_studio import project_utils

project_tags_bp = Blueprint('project_tags_bp', __name__)


@project_tags_bp.route('/<string:project>', methods=['GET'])
def tags_page(project):
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    project_config = project_utils.load_project_config(path)
    project_tags = project_config.get('project_tags', {})
    project_tags = {v: k for k, v in project_tags.items()}

    return render_template('project_tags.html', path=path, project=project, project_tags=project_tags)


@project_tags_bp.route('/create-project-tag', methods=['POST'])
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
        project_tags[tag_name] = 1

    project_config['project_tags'] = project_tags
    project_utils.write_project_config(path, project_config)
    return redirect(url_for('project_tags_bp.tags_page', project=project))


@project_tags_bp.route('/remove-project-tag', methods=['POST'])
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


@project_tags_bp.route('/edit-project-tag', methods=['POST'])
def edit_tag_in_project_tags():
    data = request.json
    path = data['path']
    tag_idx = data['tagIdx']
    new_tag_name = data['newTagName']
    project_config = project_utils.load_project_config(path)
    project_tags = project_config['project_tags']

    updated_tags = {}
    for tag_name, tag_index in project_tags.items():
        if tag_index == int(tag_idx):
            updated_tags[new_tag_name] = tag_index
        else:
            updated_tags[tag_name] = tag_index

    project_config['project_tags'] = updated_tags
    project_utils.write_project_config(path, project_config)
    return jsonify(success=True)
