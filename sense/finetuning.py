import glob
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix

from sense import camera
from sense import engine
from sense import SPLITS
from sense.engine import InferenceEngine
from sense.utils import clean_pipe_state_dict_key
from tools import directories
from tools.sense_studio import utils

MODEL_TEMPORAL_DEPENDENCY = 45
MODEL_TEMPORAL_STRIDE = 4


def set_internal_padding_false(module):
    """
    This is used to turn off padding of steppable convolution layers.
    """
    if hasattr(module, "internal_padding"):
        module.internal_padding = False


class FeaturesDataset(torch.utils.data.Dataset):
    """ Features dataset.

    This object returns a list of  features from the features dataset based on the specified parameters.

    During training, only the number of timesteps required for one temporal output is sampled from the features.

    For training with non-temporal annotations, features extracted from padded segments are discarded
    so long as the minimum video length is met.

    For training with temporal annotations, samples from the background label and non-background label
    are returned with approximately the same probability.
    """

    def __init__(self, files, labels, temporal_annotation, full_network_minimum_frames,
                 num_timesteps=None, stride=4):
        self.files = files
        self.labels = labels
        self.num_timesteps = num_timesteps
        self.stride = stride
        self.temporal_annotations = temporal_annotation
        # Compute the number of features that come from padding:
        self.num_frames_padded = int((full_network_minimum_frames - 1) / self.stride)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = np.load(self.files[idx])
        num_preds = features.shape[0]

        temporal_annotation = self.temporal_annotations[idx]
        # remove beggining of prediction that is not padded
        if temporal_annotation is not None:
            temporal_annotation = np.array(temporal_annotation)

        if self.num_timesteps and num_preds > self.num_timesteps:
            if temporal_annotation is not None:
                # creating the probability distribution based on temporal annotation
                prob0 = 1 / (2 * (np.sum(temporal_annotation == 0)))
                prob1 = 1 / (2 * (np.sum(temporal_annotation != 0)))
                probas = np.ones(len(temporal_annotation))
                probas[temporal_annotation == 0] = prob0
                probas[temporal_annotation != 0] = prob1
                probas = probas / np.sum(probas)

                # drawing the temporal label
                position = np.random.choice(len(temporal_annotation), 1, p=probas)[0]
                temporal_annotation = temporal_annotation[position:position + 1]

                # selecting the corresponding features.
                features = features[position * int(MODEL_TEMPORAL_STRIDE / self.stride): position * int(
                    MODEL_TEMPORAL_STRIDE / self.stride) + self.num_timesteps]
            else:
                # remove padded frames
                minimum_position = min(num_preds - self.num_timesteps - 1,
                                       self.num_frames_padded)
                minimum_position = max(minimum_position, 0)
                position = np.random.randint(minimum_position, num_preds - self.num_timesteps)
                features = features[position: position + self.num_timesteps]
            # will assume that we need only one output
        if temporal_annotation is None:
            temporal_annotation = [-100]
        return [features, self.labels[idx], temporal_annotation]


def generate_data_loader(project_config, features_dir, tags_dir, label_names, label2int,
                         label2int_temporal_annotation, num_timesteps=5, batch_size=16, shuffle=True,
                         stride=4, temporal_annotation_only=False,
                         full_network_minimum_frames=MODEL_TEMPORAL_DEPENDENCY):
    # Find pre-computed features and derive corresponding labels
    labels_string = []
    temporal_annotation = []

    # Use all pre-computed features
    features = []
    labels = []
    for label in label_names:
        feature_temp = glob.glob(os.path.join(features_dir, label, '*.npy'))
        features += feature_temp
        labels += [label2int[label]] * len(feature_temp)
        labels_string += [label] * len(feature_temp)

    # Check if temporal annotations exist for each video
    for label, feature in zip(labels_string, features):
        temporal_annotation_file = feature.replace(features_dir, tags_dir).replace(".npy", ".json")
        if os.path.isfile(temporal_annotation_file) and temporal_annotation_only:
            if project_config:
                tag1, tag2 = project_config['classes'][label]
            else:
                tag1 = f'{label}_tag1'
                tag2 = f'{label}_tag2'
            class_mapping = {0: 'background', 1: tag1, 2: tag2}

            annotation = json.load(open(temporal_annotation_file))["time_annotation"]
            annotation = np.array([label2int_temporal_annotation[class_mapping[y]] for y in annotation])
            temporal_annotation.append(annotation)
        else:
            temporal_annotation.append(None)

    if temporal_annotation_only:
        features = [x for x, y in zip(features, temporal_annotation) if y is not None]
        labels = [x for x, y in zip(labels, temporal_annotation) if y is not None]
        temporal_annotation = [x for x in temporal_annotation if x is not None]

    # Build data-loader
    dataset = FeaturesDataset(features, labels, temporal_annotation,
                              num_timesteps=num_timesteps, stride=stride,
                              full_network_minimum_frames=full_network_minimum_frames)
    try:
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    except ValueError:
        # The project is temporal, but annotations do not exist for train or valid.
        return None


def extract_frames(video_path, inference_engine, path_frames=None, return_frames=True):
    save_frames = path_frames is not None and not os.path.exists(path_frames)

    if not save_frames and not return_frames:
        # Nothing to do
        return None

    # Read frames from video
    video_source = camera.VideoSource(size=inference_engine.expected_frame_size,
                                      filename=video_path,
                                      target_fps=inference_engine.fps)
    frames = []

    while True:
        images = video_source.get_image()
        if images is None:
            break
        else:
            image, image_rescaled = images
            frames.append(image_rescaled)

    frames = np.array(frames)

    # Save frames if a path was provided
    if save_frames:
        os.makedirs(path_frames)

        for idx, frame in enumerate(frames[::MODEL_TEMPORAL_STRIDE]):
            Image.fromarray(frame[:, :, ::-1]).resize((400, 300)).save(
                os.path.join(path_frames, f'{idx}.jpg'), quality=50)

    return frames


def compute_features(path_features, inference_engine, frames, batch_size=None, num_timesteps=1):
    # Compute how many frames are padded to the left in order to "warm up" the model -- removing previous predictions
    # from the internal states -- with the first image, and to ensure we have enough frames in the video.
    # We also want the first non padding frame to output a feature
    frames_to_add = MODEL_TEMPORAL_STRIDE * (MODEL_TEMPORAL_DEPENDENCY // MODEL_TEMPORAL_STRIDE + 1) - 1

    # Possible improvement: investigate if a symmetric or reflect padding could be better for
    # temporal annotation prediction instead of the static first frame
    frames = np.pad(frames, ((frames_to_add, 0), (0, 0), (0, 0), (0, 0)), mode='edge')
    clip = frames[None].astype(np.float32)

    # Run the model on padded frames in order to remove the state in the current model coming
    # from the previous video.
    pre_features = inference_engine.infer(clip[:, 0:frames_to_add + 1], batch_size=batch_size)

    # Depending on the number of layers we finetune, we keep the number of features from padding
    # equal to the temporal dependency of the model.
    temporal_dependency_features = np.array(pre_features)[-num_timesteps:]

    # Predictions of the actual video frames
    predictions = inference_engine.infer(clip[:, frames_to_add + 1:], batch_size=batch_size)
    predictions = np.concatenate([temporal_dependency_features, predictions], axis=0)
    features = np.array(predictions)

    # Save features
    os.makedirs(os.path.dirname(path_features), exist_ok=True)
    np.save(path_features, features)


def compute_frames_and_features(inference_engine: InferenceEngine, project_path: str, videos_dir: str,
                                frames_dir: str, features_dir: str):
    """
    Split the videos in the given directory into frames and compute features on each frame.
    Results are stored in the given directories for frames and features.

    :param inference_engine:
        Initialized InferenceEngine that can be used for computing the features.
    :param project_path:
        The path of the current project.
    :param videos_dir:
        Directory where the videos are stored.
    :param frames_dir:
        Directory where frames should be stored. One sub-directory will be created per video with extracted frames as
        numbered .jpg files in there.
    :param features_dir:
        Directory where computed features should be stored. One .npy file will be created per video.
    """
    # Create features and frames folders
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Loop through all videos for the given class-label
    videos = glob.glob(os.path.join(videos_dir, '*.mp4'))
    num_videos = len(videos)
    for idx, video_path in enumerate(videos):
        print(f'\r  {videos_dir}  -->  Processing video {idx + 1} / {num_videos}',
              end='' if idx < (num_videos - 1) else '\n')

        video_name = os.path.basename(video_path).replace('.mp4', '')
        path_frames = os.path.join(frames_dir, video_name)
        path_features = os.path.join(features_dir, f'{video_name}.npy')

        features_needed = (utils.get_project_setting(project_path, 'assisted_tagging')
                           and not os.path.exists(path_features))

        frames = extract_frames(video_path=video_path,
                                inference_engine=inference_engine,
                                path_frames=path_frames,
                                return_frames=features_needed)

        if features_needed:
            compute_features(path_features=path_features,
                             inference_engine=inference_engine,
                             frames=frames,
                             batch_size=64,
                             num_timesteps=1)


def extract_features(path_in, model_config, net, num_layers_finetune, use_gpu, num_timesteps=1, log_fn=print):
    # Create inference engine
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)

    # extract features
    for split in SPLITS:
        videos_dir = directories.get_videos_dir(path_in, split)
        features_dir = directories.get_features_dir(path_in, split, model_config, num_layers_finetune)
        video_files = glob.glob(os.path.join(videos_dir, "*", "*.mp4"))

        num_videos = len(video_files)
        log_fn(f"\nFound {num_videos} videos to process in the {split}-set")
        for video_index, video_path in enumerate(video_files):
            log_fn(f'\rExtract features from video {video_index + 1} / {num_videos}')
            path_features = video_path.replace(videos_dir, features_dir).replace(".mp4", ".npy")

            if os.path.isfile(path_features):
                log_fn("\tSkipped - feature was already precomputed.")
            else:
                # Read all frames
                frames = extract_frames(video_path=video_path,
                                        inference_engine=inference_engine)
                compute_features(path_features=path_features,
                                 inference_engine=inference_engine,
                                 frames=frames,
                                 batch_size=16,
                                 num_timesteps=num_timesteps)

        log_fn('\n')


def training_loops(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule, label_names, label_names_temporal,
                   path_out, temporal_annotation_training=False, log_fn=print, confmat_event=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_state_dict = None
    best_top1 = -1.
    best_loss = float('inf')

    for epoch in range(0, num_epochs):  # loop over the dataset multiple times
        new_lr = lr_schedule.get(epoch)
        if new_lr:
            log_fn(f"update lr to {new_lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        net.train()
        train_loss, train_top1, cnf_matrix = run_epoch(train_loader, net, criterion, label_names_temporal,
                                                       optimizer, use_gpu,
                                                       temporal_annotation_training=temporal_annotation_training)
        net.eval()
        valid_loss, valid_top1, cnf_matrix = run_epoch(valid_loader, net, criterion, label_names_temporal,
                                                       None, use_gpu,
                                                       temporal_annotation_training=temporal_annotation_training)

        log_fn('[%d] train loss: %.3f train top1: %.3f valid loss: %.3f top1: %.3f'
               % (epoch + 1, train_loss, train_top1, valid_loss, valid_top1))

        if not temporal_annotation_training:
            if valid_top1 > best_top1:
                best_top1 = valid_top1
                best_state_dict = net.state_dict().copy()
                save_confusion_matrix(path_out, cnf_matrix, label_names, confmat_event=confmat_event)
        else:
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state_dict = net.state_dict().copy()
                save_confusion_matrix(path_out, cnf_matrix, label_names_temporal, confmat_event=confmat_event)

        # save the last checkpoint
        model_state_dict = net.state_dict().copy()
        model_state_dict = {clean_pipe_state_dict_key(key): value
                            for key, value in model_state_dict.items()}
        torch.save(model_state_dict, os.path.join(path_out, "last_classifier.checkpoint"))

    log_fn('Finished Training')
    return best_state_dict


def run_epoch(data_loader, net, criterion, label_names_temporal, optimizer=None, use_gpu=False,
              temporal_annotation_training=False):
    running_loss = 0.0
    epoch_top_predictions = []
    epoch_labels = []

    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, targets]
        inputs, targets, temporal_annotation = data
        if temporal_annotation_training:
            targets = temporal_annotation
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward + backward + optimize
        if net.training:
            # Run on each batch element independently
            outputs = [net(input_i) for input_i in inputs]
            # Concatenate outputs to get a tensor of size batch_size x num_classes
            outputs = torch.cat(outputs, dim=0)

            if temporal_annotation_training:
                # take only targets one batch
                targets = targets[:, 0]
                # realign the number of outputs
                min_pred_number = min(outputs.shape[0], targets.shape[0])
                targets = targets[0:min_pred_number]
                outputs = outputs[0:min_pred_number]
        else:
            # Average predictions on the time dimension to get a tensor of size 1 x num_classes
            # This assumes validation operates with batch_size=1 and process all available features (no cropping)
            assert data_loader.batch_size == 1
            outputs = net(inputs[0])
            if temporal_annotation_training:
                targets = targets[0]
                # realign the number of outputs
                min_pred_number = min(outputs.shape[0], targets.shape[0])
                targets = targets[0:min_pred_number]
                outputs = outputs[0:min_pred_number]
            else:
                outputs = torch.mean(outputs, dim=0, keepdim=True)

        loss = criterion(outputs, targets)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Store label and predictions to compute statistics later
        epoch_labels += list(targets.cpu().numpy())
        epoch_top_predictions += list(outputs.argmax(dim=1).cpu().numpy())

        # print statistics
        running_loss += loss.item()

    epoch_labels = np.array(epoch_labels)
    epoch_top_predictions = np.array(epoch_top_predictions)

    top1 = np.mean(epoch_labels == epoch_top_predictions)
    loss = running_loss / len(data_loader)

    if temporal_annotation_training:
        cnf_matrix = confusion_matrix(epoch_labels, epoch_top_predictions, labels=range(0, len(label_names_temporal)))
    else:
        cnf_matrix = confusion_matrix(epoch_labels, epoch_top_predictions)

    return loss, top1, cnf_matrix


def save_confusion_matrix(
        path_out,
        confusion_matrix_array,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues,
        confmat_event=None,
):
    """
    This function creates a matplotlib figure out of the provided confusion matrix and saves it
    to a file. The provided numpy array is also saved. Normalization can be applied by setting
    `normalize=True`.
    """

    plt.figure()
    plt.imshow(confusion_matrix_array, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    accuracy = np.diag(confusion_matrix_array).sum() / confusion_matrix_array.sum()
    title += '\nAccuracy={:.1f}'.format(100 * float(accuracy))
    plt.title(title)

    if normalize:
        confusion_matrix_array = confusion_matrix_array.astype('float')
        confusion_matrix_array /= confusion_matrix_array.sum(axis=1)[:, np.newaxis]

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = confusion_matrix_array.max() / 2.
    for i, j in itertools.product(range(confusion_matrix_array.shape[0]),
                                  range(confusion_matrix_array.shape[1])):
        plt.text(j, i, confusion_matrix_array[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix_array[i, j] > thresh else "black")

    plt.savefig(os.path.join(path_out, 'confusion_matrix.png'), bbox_inches='tight',
                transparent=False, pad_inches=0.1, dpi=300)
    plt.close()

    np.save(os.path.join(path_out, 'confusion_matrix.npy'), confusion_matrix_array)

    if confmat_event is not None:
        confmat_event.set()
