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


def _backwards_compatibility_update(path, config):
    updated = False

    if 'use_gpu' not in config:
        config['use_gpu'] = False
        updated = True

    if 'temporal' not in config:
        config['temporal'] = False
        updated = True

    if 'assisted_tagging' not in config:
        config['assisted_tagging'] = False
        updated = True

    if 'video_recording' not in config:
        config['video_recording'] = {
            'countdown': 3,
            'recording': 5,
        }
        updated = True

    if 'tags' not in config:
        # Collect class-wise tags
        old_classes = config['classes']
        tags_list = []
        for class_name, class_tags in old_classes.items():
            tags_list.extend(class_tags)

        # Assign project-wide unique indices to tags (0 is reserved for 'background')
        tags = {idx + 1: tag_name for idx, tag_name in enumerate(sorted(tags_list))}
        config['tags'] = tags
        config['max_tag_index'] = len(tags_list)

        # Setup class dictionary with tag indices
        inverse_tags = {tag_name: tag_idx for tag_idx, tag_name in tags.items()}
        inverse_tags['background'] = 0
        config['classes'] = {
            class_name: [inverse_tags[tag_name] for tag_name in class_tags]
            for class_name, class_tags in old_classes.items()
        }

        # Translate existing annotations
        for split in SPLITS:
            for label, label_tags in old_classes.items():
                tags_dir = directories.get_tags_dir(path, split, label)
                if os.path.exists(tags_dir):
                    label_tags = ['background'] + label_tags

                    for video_name in os.listdir(tags_dir):
                        annotation_file = os.path.join(tags_dir, video_name)
                        with open(annotation_file, 'r') as f:
                            annotation_data = json.load(f)

                        # Translate relative indices [0, 1, 2] to their names and then to their new absolute indices
                        new_annotations = [inverse_tags[label_tags[idx]] for idx in annotation_data['time_annotation']]
                        annotation_data['time_annotation'] = new_annotations

                        with open(annotation_file, 'w') as f:
                            json.dump(annotation_data, f, indent=2)

        updated = True
    else:
        # Translate string keys to integers (because JSON does not store integer keys)
        config['tags'] = {int(idx_str): tag_name for idx_str, tag_name in config['tags'].items()}

    if updated:
        # Save updated config
        write_project_config(path, config)

    return config


def load_project_config(path):
    config_path = os.path.join(path, PROJECT_CONFIG_FILE)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        config = _backwards_compatibility_update(path, config)
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
            'tags': {},
            'max_tag_index': 0,
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
    return config[setting]


def toggle_project_setting(path, setting):
    config = load_project_config(path)
    current_status = config[setting]

    new_status = not current_status
    config[setting] = new_status
    write_project_config(path, config)

    return new_status


def get_timer_default(path):
    """Get the default countdown and recording duration (in seconds) for video-recording."""
    config = load_project_config(path)
    countdown = config['video_recording']['countdown']
    duration = config['video_recording']['recording']

    return countdown, duration


def set_timer_default(path, countdown, recording):
    """Set the new default countdown and recording duration (in seconds) for video-recording."""
    config = load_project_config(path)
    video_recording = config['video_recording']

    video_recording['countdown'] = countdown
    video_recording['recording'] = recording
    config['video_recording'] = video_recording

    write_project_config(path, config)
