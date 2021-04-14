import datetime
import json
import os

from sense import SPLITS
from tools import directories

MODULE_DIR = os.path.dirname(__file__)
PROJECTS_OVERVIEW_CONFIG_FILE = os.path.join(MODULE_DIR, 'projects_config.json')

PROJECT_CONFIG_FILE = 'project_config.json'


def load_project_overview_config():
    if os.path.isfile(PROJECTS_OVERVIEW_CONFIG_FILE):
        with open(PROJECTS_OVERVIEW_CONFIG_FILE, 'r') as f:
            projects = json.load(f)
        return projects
    else:
        write_project_overview_config({})
        return {}


def write_project_overview_config(projects):
    with open(PROJECTS_OVERVIEW_CONFIG_FILE, 'w') as f:
        json.dump(projects, f, indent=2)


def lookup_project_path(project_name):
    projects = load_project_overview_config()
    return projects[project_name]['path']


def load_project_config(path):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = None
    return config


def write_project_config(path, config):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def setup_new_project(project_name, path, config=None):
    """
    Setup a project directory with a config file and the directories for train and valid videos and
    add an entry to the projects overview config.
    If an existing project config is given, this one will be used and the project name in there will
    be updated.
    """
    if not config:
        # Setup new project config
        config = {
            'name': project_name,
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
    else:
        config['name'] = project_name

    write_project_config(path, config)

    # Setup directory structure
    for split in SPLITS:
        videos_dir = directories.get_videos_dir(path, split)
        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

    # Update overall projects config file
    projects = load_project_overview_config()
    projects[project_name] = {
        'path': path,
    }

    write_project_overview_config(projects)

    return config


def get_folder_name_for_project(project_name):
    """
    Construct a folder name from the given project name by converting to lower case and replacing
    spaces with underscores: My Project -> my_project
    """
    return project_name.lower().replace(' ', '_')


def get_unique_project_name(base_name):
    """
    Make the given project name unique by adding a suffix such as "(2)" if necessary.
    """
    projects = load_project_overview_config()
    project_name = base_name
    idx = 2
    while project_name in projects:
        project_name = f'{base_name} ({idx})'
        idx += 1

    return project_name


def get_project_setting(path, setting):
    config = load_project_config(path)
    return config.get(setting, False)


def toggle_project_setting(path, setting):
    config = load_project_config(path)
    current_status = config.get(setting, False)

    new_status = not current_status
    config[setting] = new_status
    write_project_config(path, config)

    return new_status


def get_timer_default(path):
    """Get the default countdown and recording duration (in seconds) for video-recording."""
    config = load_project_config(path)
    countdown = config.get('video_recording', {}).get('countdown', 3)
    duration = config.get('video_recording', {}).get('recording', 5)

    return countdown, duration


def set_timer_default(path, countdown, recording):
    """Set the new default countdown and recording duration (in seconds) for video-recording."""
    config = load_project_config(path)
    video_recording = config.get('video_recording', {})

    video_recording['countdown'] = countdown
    video_recording['recording'] = recording
    config['video_recording'] = video_recording

    write_project_config(path, config)
