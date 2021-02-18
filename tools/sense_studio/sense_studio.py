#!/usr/bin/env python
"""
Web app for maintaining all of your video datasets:
- Setup new datasets with custom labels and temporal tags
- Record new videos (coming soon)
- Temporally annotate your videos with custom tags
- Train custom models using strong backbone networks (coming soon)
"""

import datetime
import glob
import os
import urllib

from flask import Flask
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for

from annotations.annotations import annotations_bp
from annotations.prepare_annotations import prepare_annotations_bp
from video_recorder.record_videos import video_recorder_bp

from tools.sense_studio.utils import _load_project_config
from tools.sense_studio.utils import _load_project_overview_config
from tools.sense_studio.utils import _lookup_project_path
from tools.sense_studio.utils import SPLITS
from tools.sense_studio.utils import _write_project_config
from tools.sense_studio.utils import _write_project_overview_config
from tools.sense_studio.utils import _get_class_name_and_tags

app = Flask(__name__)
app.secret_key = 'd66HR8dç"f_-àgjYYic*dh'

app.register_blueprint(prepare_annotations_bp, url_prefix='/prepare-annotation')
app.register_blueprint(annotations_bp, url_prefix='/annotate')
app.register_blueprint(video_recorder_bp, url_prefix='/video-recorder')


@app.route('/')
def projects_overview():
    """
    Home page of SenseStudio. Show the overview of all registered projects and check if their
    locations are still valid.
    """
    projects = _load_project_overview_config()

    # Check if project paths still exist
    for name, project in projects.items():
        project['exists'] = os.path.exists(project['path'])

    return render_template('projects_overview.html', projects=projects)


@app.route('/projects-list', methods=['POST'])
def projects_list():
    """
    Provide the current list of projects to external callers.
    """
    projects = _load_project_overview_config()
    return jsonify(projects)


@app.route('/project-config', methods=['POST'])
def project_config():
    """
    Provide the config for a given project.
    """
    data = request.json
    name = data['name']
    path = _lookup_project_path(name)

    # Get config
    config = _load_project_config(path)
    return jsonify(config)


@app.route('/remove-project/<string:name>')
def remove_project(name):
    """
    Remove a given project from the config file and reload the overview page.
    """
    name = urllib.parse.unquote(name)
    projects = _load_project_overview_config()

    del projects[name]

    _write_project_overview_config(projects)

    return redirect(url_for('projects_overview'))


@app.route('/browse-directory', methods=['POST'])
def browse_directory():
    """
    Browse the local file system starting at the given path and provide the following information:
    - path_exists: If the given path exists
    - subdirs: The list of sub-directories at the given path
    """
    data = request.json
    path = data['path']

    subdirs = [d for d in glob.glob(f'{path}*') if os.path.isdir(d)] if os.path.isabs(path) else []

    return jsonify(path_exists=os.path.exists(path), subdirs=subdirs)


@app.route('/setup-project', methods=['POST'])
def setup_project():
    """
    Add a new project to the config file. Can also be used for updating an existing project.
    """
    data = request.form
    name = data['projectName']
    path = data['path']

    # Initialize project directory
    if not os.path.exists(path):
        os.mkdir(path)

    # Update project config
    try:
        # Check for existing config file
        config = _load_project_config(path)
        old_name = config['name']
        config['name'] = name
    except FileNotFoundError:
        # Setup new project config
        config = {
            'name': name,
            'date_created': datetime.date.today().isoformat(),
            'classes': {},
        }
        old_name = None

    _write_project_config(path, config)

    # Setup directory structure
    for split in SPLITS:
        videos_dir = os.path.join(path, f'videos_{split}')
        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

    # Update overall projects config file
    projects = _load_project_overview_config()

    if old_name and old_name in projects:
        del projects[old_name]

    projects[name] = {
        'path': path,
    }

    _write_project_overview_config(projects)

    return redirect(url_for('project_details', project=name))


@app.route('/project/<string:project>')
def project_details(project):
    """
    Show the details for the selected project.
    """
    project = urllib.parse.unquote(project)
    path = _lookup_project_path(project)
    config = _load_project_config(path)

    stats = {}
    for class_name, tags in config['classes'].items():
        stats[class_name] = {}
        for split in SPLITS:
            videos_path = os.path.join(path, f'videos_{split}', class_name)
            tags_path = os.path.join(path, f'tags_{split}', class_name)
            stats[class_name][split] = {
                'total': len(os.listdir(videos_path)),
                'tagged': len(os.listdir(tags_path)) if os.path.exists(tags_path) else 0,
            }

    return render_template('project_details.html', config=config, project=project, path=path, stats=stats)


@app.route('/add-class/<string:project>', methods=['POST'])
def add_class(project):
    """
    Add a new class to the given project.
    """
    project = urllib.parse.unquote(project)
    path = _lookup_project_path(project)

    # Get class name and tags
    class_name, tag1, tag2 = _get_class_name_and_tags(request.form)

    # Update project config
    config = _load_project_config(path)
    config['classes'][class_name] = [tag1, tag2]
    _write_project_config(path, config)

    # Setup directory structure
    for split in SPLITS:
        videos_dir = os.path.join(path, f'videos_{split}')
        class_dir = os.path.join(videos_dir, class_name)

        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

    return redirect(url_for("project_details", project=project))


@app.route('/edit-class/<string:project>/<string:class_name>', methods=['POST'])
def edit_class(project, class_name):
    """
    Edit the class name and tags for an existing class in the given project.
    """
    project = urllib.parse.unquote(project)
    class_name = urllib.parse.unquote(class_name)
    path = _lookup_project_path(project)

    # Get new class name and tags
    new_class_name, new_tag1, new_tag2 = _get_class_name_and_tags(request.form)

    # Update project config
    config = _load_project_config(path)
    del config['classes'][class_name]
    config['classes'][new_class_name] = [new_tag1, new_tag2]
    _write_project_config(path, config)

    # Update directory names
    prefixes = ['videos', 'features', 'frames', 'tags']
    for split in SPLITS:
        for prefix in prefixes:
            main_dir = os.path.join(path, f'{prefix}_{split}')
            class_dir = os.path.join(main_dir, class_name)

            if os.path.exists(class_dir):
                new_class_dir = os.path.join(main_dir, new_class_name)
                os.rename(class_dir, new_class_dir)

    logreg_dir = os.path.join(path, 'logreg')
    class_dir = os.path.join(logreg_dir, class_name)

    if os.path.exists(class_dir):
        new_class_dir = os.path.join(logreg_dir, new_class_name)
        os.rename(class_dir, new_class_dir)

    return redirect(url_for("project_details", project=project))


@app.route('/remove-class/<string:project>/<string:class_name>')
def remove_class(project, class_name):
    """
    Remove the given class from the config file of the given project. No data will be deleted.
    """
    project = urllib.parse.unquote(project)
    class_name = urllib.parse.unquote(class_name)
    path = _lookup_project_path(project)

    # Update project config
    config = _load_project_config(path)
    del config['classes'][class_name]
    _write_project_config(path, config)

    return redirect(url_for("project_details", project=project))


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(debug=True)
