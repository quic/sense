import glob
import json
import numpy as np
import os
import urllib

from flask import Blueprint
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import url_for
from joblib import dump
from joblib import load
from natsort import natsorted
from natsort import ns
from sklearn.linear_model import LogisticRegression

from sense import SPLITS
from sense.finetuning import compute_frames_and_features
from tools import directories
from tools.sense_studio import project_utils
from tools.sense_studio import utils


annotation_bp = Blueprint('annotation_bp', __name__)


@annotation_bp.route('/<string:project>/<string:split>/<string:label>')
def show_video_list(project, split, label):
    """
    Show the list of videos for the given split, class label and project.
    If the necessary files for annotation haven't been prepared yet, this is done now.
    """
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    split = urllib.parse.unquote(split)
    label = urllib.parse.unquote(label)

    # load feature extractor
    inference_engine, model_config = utils.load_feature_extractor(path)

    videos_dir = directories.get_videos_dir(path, split, label)
    frames_dir = directories.get_frames_dir(path, split, label)
    features_dir = directories.get_features_dir(path, split, model_config, label=label)
    tags_dir = directories.get_tags_dir(path, split, label)

    os.makedirs(tags_dir, exist_ok=True)

    # compute the features and frames missing
    compute_frames_and_features(inference_engine=inference_engine,
                                project_path=path,
                                videos_dir=videos_dir,
                                frames_dir=frames_dir,
                                features_dir=features_dir)

    videos = os.listdir(frames_dir)
    videos = natsorted(videos, alg=ns.IC)

    tagged_list = set(os.listdir(tags_dir))
    tagged = [f'{video}.json' in tagged_list for video in videos]

    num_videos = len(videos)
    num_tagged = len(tagged_list)
    num_untagged = num_videos - num_tagged

    video_list = zip(videos, tagged, list(range(len(videos))))
    return render_template('video_list.html', video_list=video_list, split=split, label=label, path=path,
                           project=project, num_videos=num_videos, num_tagged=num_tagged, num_untagged=num_untagged)


@annotation_bp.route('/<string:project>/<string:split>/<string:label>/<int:idx>')
def annotate(project, split, label, idx):
    """
    For the given class label, show all frames for annotating the selected video.
    """
    project = urllib.parse.unquote(project)
    path = project_utils.lookup_project_path(project)
    label = urllib.parse.unquote(label)
    split = urllib.parse.unquote(split)

    config = project_utils.load_project_config(path)
    tags = config['tags'].copy()
    tags[0] = 'background'

    class_tags = config['classes'][label].copy()
    class_tags.append(0)  # Always add 'background'
    class_tags.sort()

    _, model_config = utils.load_feature_extractor(path)

    frames_dir = directories.get_frames_dir(path, split, label)
    features_dir = directories.get_features_dir(path, split, model_config, label=label)
    tags_dir = directories.get_tags_dir(path, split, label)
    logreg_dir = directories.get_logreg_dir(path, model_config)

    videos = os.listdir(frames_dir)
    videos = natsorted(videos, alg=ns.IC)

    # The list of images in the folder
    images = [image for image in glob.glob(os.path.join(frames_dir, videos[idx], '*'))
              if utils.is_image_file(image)]
    classes = [-1] * len(images)

    # Load logistic regression model if available and assisted tagging is enabled
    if utils.get_project_setting(path, 'assisted_tagging'):
        logreg_path = os.path.join(logreg_dir, 'logreg.joblib')
        features_path = os.path.join(features_dir, f'{videos[idx]}.npy')
        if os.path.isfile(logreg_path) and os.path.isfile(features_path):
            logreg = load(logreg_path)
            features = np.load(features_path).mean(axis=(2, 3))
            classes = list(logreg.predict(features))

            # Reset tags that have been removed from the class to 'background'
            classes = [tag_idx if tag_idx in class_tags else 0 for tag_idx in classes]

    # Natural sort images, so that they are sorted by number
    images = natsorted(images, alg=ns.IC)
    # Extract image file name (without full path) and include class label
    images = [(os.path.basename(image), _class) for image, _class in zip(images, classes)]

    # Load existing annotations
    annotations_file = os.path.join(tags_dir, f'{videos[idx]}.json')
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            annotations = data['time_annotation']

            # Reset tags that have been removed from the class to 'background'
            annotations = [tag_idx if tag_idx in class_tags else 0 for tag_idx in annotations]
    else:
        # Use "background" label for all frames per default
        annotations = [0] * len(images)

    return render_template('frame_annotation.html', images=images, annotations=annotations, idx=idx, fps=16,
                           n_images=len(images), video_name=videos[idx], project_config=config,
                           split=split, label=label, path=path, project=project, n_videos=len(videos),
                           tags=tags, class_tags=class_tags)


@annotation_bp.route('/submit-annotation', methods=['POST'])
def submit_annotation():
    """
    Submit annotated tags for all frames and save them to a json file.
    """
    data = request.form  # a multi-dict containing POST data
    idx = int(data['idx'])
    fps = float(data['fps'])
    path = data['path']
    project = data['project']
    split = data['split']
    label = data['label']
    video = data['video']
    next_frame_idx = idx + 1

    frames_dir = directories.get_frames_dir(path, split, label)
    tags_dir = directories.get_tags_dir(path, split, label)
    description = {'file': f'{video}.mp4', 'fps': fps}

    out_annotation = os.path.join(tags_dir, f'{video}.json')
    time_annotation = []

    for frame_idx in range(int(data['n_images'])):
        time_annotation.append(int(data[f'{frame_idx}_tag']))

    description['time_annotation'] = time_annotation

    with open(out_annotation, 'w') as f:
        json.dump(description, f, indent=2)

    # Automatic re-training of the logistic regression model
    if utils.get_project_setting(path, 'assisted_tagging'):
        train_logreg(path=path)

    if next_frame_idx >= len(os.listdir(frames_dir)):
        return redirect(url_for('.show_video_list', project=project, split=split, label=label))

    return redirect(url_for('.annotate', split=split, label=label, project=project, idx=next_frame_idx))


@annotation_bp.route('/uploads/<string:project>/<string:split>/<string:label>/<string:video_name>/<string:img_file>')
def download_file(project, split, label, video_name, img_file):
    """
    Load an image from the given path.
    """
    dataset_path = project_utils.lookup_project_path(project)
    img_dir = os.path.join(directories.get_frames_dir(dataset_path, split, label), video_name)
    return send_from_directory(img_dir, img_file, as_attachment=True)


def train_logreg(path):
    """
    (Re-)Train a logistic regression model on all annotations that have been submitted so far.
    """
    inference_engine, model_config = utils.load_feature_extractor(path)

    logreg_dir = directories.get_logreg_dir(path, model_config)
    logreg_path = os.path.join(logreg_dir, 'logreg.joblib')
    project_config = project_utils.load_project_config(path)
    classes = project_config['classes']

    all_features = []
    all_annotations = []

    for split in SPLITS:
        for label, class_tags in classes.items():
            videos_dir = directories.get_videos_dir(path, split, label)
            frames_dir = directories.get_frames_dir(path, split, label)
            features_dir = directories.get_features_dir(path, split, model_config, label=label)
            tags_dir = directories.get_tags_dir(path, split, label)

            if not os.path.exists(tags_dir):
                continue

            # Compute the respective frames and features
            compute_frames_and_features(inference_engine=inference_engine,
                                        project_path=path,
                                        videos_dir=videos_dir,
                                        frames_dir=frames_dir,
                                        features_dir=features_dir)

            video_tag_files = os.listdir(tags_dir)

            for video_tag_file in video_tag_files:
                feature_file = os.path.join(features_dir, video_tag_file.replace('.json', '.npy'))
                annotation_file = os.path.join(tags_dir, video_tag_file)

                features = np.load(feature_file)
                for f in features:
                    all_features.append(f.mean(axis=(1, 2)))

                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)['time_annotation']

                # Reset tags that have been removed from the class to 'background'
                annotations = [tag_idx if tag_idx in class_tags else 0 for tag_idx in annotations]

                all_annotations.extend(annotations)

    # Use low class weight for background and higher weight for all present tags
    annotated_tags = set(all_annotations)
    class_weight = {tag: 2 for tag in annotated_tags}
    class_weight[0] = 0.5

    all_features = np.array(all_features)
    all_annotations = np.array(all_annotations)

    if len(annotated_tags) > 1:
        os.makedirs(logreg_dir, exist_ok=True)
        logreg = LogisticRegression(C=0.1, class_weight=class_weight)
        logreg.fit(all_features, all_annotations)
        dump(logreg, logreg_path)
