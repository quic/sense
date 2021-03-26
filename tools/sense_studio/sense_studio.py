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

from sense import SPLITS
from sense.finetuning import compute_frames_and_features
from tools import directories
from tools.sense_studio import utils
from tools.sense_studio.annotation import annotation_bp
from tools.sense_studio.video_recording import video_recording_bp

app = Flask(__name__)
app.secret_key = 'd66HR8dç"f_-àgjYYic*dh'

app.register_blueprint(annotation_bp, url_prefix='/annotation')
app.register_blueprint(video_recording_bp, url_prefix='/video-recording')


@app.route('/')
def projects_overview():
    """
    Home page of SenseStudio. Show the overview of all registered projects and check if their
    locations are still valid.
    """
    projects = utils.load_project_overview_config()

    # Check if project paths still exist
    for name, project in projects.items():
        project['exists'] = os.path.exists(project['path'])

    return render_template('projects_overview.html', projects=projects)


@app.route('/projects-list')
def projects_list():
    """
    Provide the current list of projects to external callers.
    """
    projects = utils.load_project_overview_config()
    return jsonify(projects)


@app.route('/project-config', methods=['POST'])
def project_config():
    """
    Provide the config for a given project.
    """
    data = request.json
    name = data['name']
    path = utils.lookup_project_path(name)

    # Get config
    config = utils.load_project_config(path)
    return jsonify(config)


@app.route('/remove-project/<string:name>')
def remove_project(name):
    """
    Remove a given project from the config file and reload the overview page.
    """
    name = urllib.parse.unquote(name)
    projects = utils.load_project_overview_config()

    del projects[name]

    utils.write_project_overview_config(projects)

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
        config = utils.load_project_config(path)
        old_name = config['name']
        config['name'] = name
    except FileNotFoundError:
        # Setup new project config
        config = {
            'name': name,
            'date_created': datetime.date.today().isoformat(),
            'classes': {},
            'use_gpu': False,
            'temporal': False,
            'assisted_tagging': False,
            'video_recording': {
                'countdown': 3,
                'recording': 5,
            },
        }
        old_name = None

    utils.write_project_config(path, config)

    # Setup directory structure
    for split in SPLITS:
        videos_dir = directories.get_videos_dir(path, split)
        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

    # Update overall projects config file
    projects = utils.load_project_overview_config()

    if old_name and old_name in projects:
        del projects[old_name]

    projects[name] = {
        'path': path,
    }

    utils.write_project_overview_config(projects)

    return redirect(url_for('project_details', project=name))


@app.route('/project/<string:project>')
def project_details(project):
    """
    Show the details for the selected project.
    """
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)
    config = utils.load_project_config(path)

    stats = {}
    for class_name, tags in config['classes'].items():
        stats[class_name] = {}
        for split in SPLITS:
            videos_dir = directories.get_videos_dir(path, split, class_name)
            tags_dir = directories.get_tags_dir(path, split, class_name)
            stats[class_name][split] = {
                'total': len(os.listdir(videos_dir)),
                'tagged': len(os.listdir(tags_dir)) if os.path.exists(tags_dir) else 0,
            }

    return render_template('project_details.html', config=config, path=path, stats=stats, project=config['name'])


@app.route('/add-class/<string:project>', methods=['POST'])
def add_class(project):
    """
    Add a new class to the given project.
    """
    project = urllib.parse.unquote(project)
    path = utils.lookup_project_path(project)

    # Get class name and tags
    class_name, tag1, tag2 = utils.get_class_name_and_tags(request.form)

    # Update project config
    config = utils.load_project_config(path)
    config['classes'][class_name] = [tag1, tag2]
    utils.write_project_config(path, config)

    # Setup directory structure
    for split in SPLITS:
        videos_dir = directories.get_videos_dir(path, split, class_name)

        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

    return redirect(url_for("project_details", project=project))


@app.route('/toggle-project-setting', methods=['POST'])
def toggle_project_setting():
    """
    Toggle boolean project setting.
    """
    data = request.json
    path = data['path']
    setting = data['setting']
    new_status = utils.toggle_project_setting(path, setting)

    # Update logreg model if assisted tagging was just enabled
    if setting == 'assisted_tagging' and new_status:
        split = data['split']
        label = data['label']
        inference_engine, model_config = utils.load_feature_extractor(path)

        videos_dir = directories.get_videos_dir(path, split, label)
        frames_dir = directories.get_frames_dir(path, split, label)
        features_dir = directories.get_features_dir(path, split, model_config, label=label)

        # Compute the respective frames and features
        compute_frames_and_features(inference_engine=inference_engine,
                                    project_path=path,
                                    videos_dir=videos_dir,
                                    frames_dir=frames_dir,
                                    features_dir=features_dir)

        # Re-train the logistic regression model
        utils.train_logreg(path=path, split=split, label=label)

    return jsonify(setting_status=new_status)


@app.route('/edit-class/<string:project>/<string:class_name>', methods=['POST'])
def edit_class(project, class_name):
    """
    Edit the class name and tags for an existing class in the given project.
    """
    project = urllib.parse.unquote(project)
    class_name = urllib.parse.unquote(class_name)
    path = utils.lookup_project_path(project)

    # Get new class name and tags
    new_class_name, new_tag1, new_tag2 = utils.get_class_name_and_tags(request.form)

    # Update project config
    config = utils.load_project_config(path)
    del config['classes'][class_name]
    config['classes'][new_class_name] = [new_tag1, new_tag2]
    utils.write_project_config(path, config)

    # Update directory names
    data_dirs = []
    for split in SPLITS:
        data_dirs.extend([
            directories.get_videos_dir(path, split),
            directories.get_frames_dir(path, split),
            directories.get_tags_dir(path, split),
        ])

        # Feature directories follow the format <dataset_dir>/<split>/<model>/<num_layers_to_finetune>/<label>
        features_dir = directories.get_features_dir(path, split)
        model_dirs = [os.path.join(features_dir, model_dir) for model_dir in os.listdir(features_dir)]
        data_dirs.extend([os.path.join(model_dir, tuned_layers)
                          for model_dir in model_dirs
                          for tuned_layers in os.listdir(model_dir)])

    logreg_dir = directories.get_logreg_dir(path)
    data_dirs.extend([os.path.join(logreg_dir, model_dir) for model_dir in os.listdir(logreg_dir)])

    for base_dir in data_dirs:
        class_dir = os.path.join(base_dir, class_name)

        if os.path.exists(class_dir):
            new_class_dir = os.path.join(base_dir, new_class_name)
            os.rename(class_dir, new_class_dir)

    return redirect(url_for('project_details', project=project))


@app.route('/remove-class/<string:project>/<string:class_name>')
def remove_class(project, class_name):
    """
    Remove the given class from the config file of the given project. No data will be deleted.
    """
    project = urllib.parse.unquote(project)
    class_name = urllib.parse.unquote(class_name)
    path = utils.lookup_project_path(project)

    # Update project config
    config = utils.load_project_config(path)
    del config['classes'][class_name]
    utils.write_project_config(path, config)

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


@app.context_processor
def context_processors():
    """
    This context processor will inject methods into templates,
    which can be invoked like an ordinary method in HTML templates.
    E.g. {% set project_config = inject_project_config(project) %}
    """
    def inject_project_config(project):
        path = utils.lookup_project_path(project)
        return utils.load_project_config(path)

    return dict(inject_project_config=inject_project_config)


@app.route('/set-timer-default', methods=['POST'])
def set_timer_default():
    data = request.json
    path = data['path']
    countdown = int(data['countdown'])
    recording = int(data['recording'])

    utils.set_timer_default(path, countdown, recording)

    return jsonify(status=True)


if __name__ == '__main__':
    app.run(debug=True)
