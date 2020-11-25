import os
import glob
import json

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from realtimenet import camera
from realtimenet import engine
from PIL import Image


def set_internal_padding_false(module):
    """
    This is used to turn off padding of steppable convolution layers.
    """
    if hasattr(module, "internal_padding"):
        module.internal_padding = False


class FeaturesDataset(torch.utils.data.Dataset):
    """Features dataset."""

    def __init__(self, files, labels, num_timesteps=None):
        self.files = files
        self.labels = labels
        self.num_timesteps = num_timesteps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = np.load(self.files[idx])
        num_preds = features.shape[0]
        if self.num_timesteps and num_preds > self.num_timesteps:
            position = np.random.randint(0, num_preds - self.num_timesteps)
            features = features[position: position + self.num_timesteps]
        return [features, self.labels[idx]]


class FeaturesDatasetCounting(torch.utils.data.Dataset):
    """Features dataset."""

    def __init__(self, files, labels, model_time_step,
                 num_timesteps=None, minimum_frames=45, stride=4):
        self.files = files
        self.labels = labels
        self.num_timesteps = num_timesteps
        self.stride = stride
        self.minimum_frames = minimum_frames
        self.model_time_step = model_time_step
        # besoin de prendre la dependance temporelle qui va se faire encore bouffer

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = np.load(self.files[idx])
        num_preds = features.shape[0]
        label = self.labels[idx]
        # remove beggining of prediction that is not padded
        label = label[int((self.minimum_frames - 1) /4):]
        if self.num_timesteps and num_preds > self.num_timesteps:
            position = np.random.randint(0, num_preds - self.num_timesteps)
            # compute label of the frame
            position_label = int(position / (4 / self.stride))
            features = features[position: position + self.num_timesteps]
            # will assume that we need only one output
            label = label[position_label:position_label + 1]

        # need to compute
        return [features, label]


def generate_data_loader(features_dir, label_names, label2int,
                         num_timesteps=5, batch_size=16, shuffle=True):

    # Find pre-computed features and derive corresponding labels
    features = []
    labels = []
    for label in label_names:
        feature_temp = glob.glob(f'{features_dir}/{label}/*.npy')
        features += feature_temp
        labels += [label2int[label]] * len(feature_temp)

    # Build dataloader
    dataset = FeaturesDataset(features, labels, num_timesteps=num_timesteps)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    return data_loader


def generate_data_loader_counting(dataset_dir, features_dir, tags_dir, label_names, label2int,
                         model_time_step, num_timesteps=5, batch_size=16, shuffle=True,
                                  minimum_frames=45, stride=4):

    # Find pre-computed features and derive corresponding labels
    features = []
    labels = []
    tags_dir = os.path.join(dataset_dir, tags_dir)
    features_dir = os.path.join(dataset_dir, features_dir)

    for label in label_names:
        classe_mapping = {0: "counting_background",
                          1: f'{label}_position_1', 2:
                              f'{label}_position_2'}
        feature_temp = glob.glob(f'{features_dir}/{label}/*.npy')
        feature_temp = [x for x in feature_temp if os.path.isfile(x.replace(features_dir, tags_dir).replace(".npy", ".json"))]
        tags_temp = [x.replace(features_dir, tags_dir).replace(".npy", ".json") for x in feature_temp]
        tags_temp = [json.load(open(x))["time_annotation"] for x in tags_temp]
        tags_temp = [np.array([label2int[classe_mapping[y]] for y in x]) for x in tags_temp]


        features += feature_temp
        labels += tags_temp


    # Build dataloader
    dataset = FeaturesDatasetCounting(features, labels, model_time_step=model_time_step,
                                      num_timesteps=num_timesteps, minimum_frames=minimum_frames, stride=stride)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    return data_loader


def uniform_frame_sample(video, sample_rate):
    """
    Uniformly sample video frames according to the provided sample_rate.
    """

    depth = video.shape[0]
    if sample_rate < 1.:
        indices = np.arange(0, depth, 1./sample_rate)
        offset = int((depth - indices[-1]) / 2)
        sampled_frames = (indices + offset).astype(np.int32)
        return video[sampled_frames]
    return video


def compute_features(video_path, path_out, inference_engine, minimum_frames=45, path_frames=None,
                     batch_size=None):
    video_source = camera.VideoSource(camera_id=None,
                                      size=inference_engine.expected_frame_size,
                                      filename=video_path)
    video_fps = video_source.get_fps()
    frames = []
    while True:
        images = video_source.get_image()
        if images is None:
            break
        else:
            image, image_rescaled = images
            frames.append(image_rescaled)
    frames = uniform_frame_sample(np.array(frames), inference_engine.fps / video_fps)

    if frames.shape[0] < minimum_frames:
        print(f"\nVideo too short: {video_path} - first frame will be duplicated")
        num_missing_frames = minimum_frames - frames.shape[0]
        frames = np.pad(frames, ((num_missing_frames, 0), (0, 0), (0, 0), (0, 0)),
                        mode='edge')
    # Inference
    clip = frames[None].astype(np.float32)
    predictions = inference_engine.infer(clip, batch_size=batch_size)
    features = np.array(predictions)
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    np.save(path_out, features)
    if path_frames is not None:
        os.makedirs(os.path.dirname(path_frames), exist_ok=True)
        frames_to_save = []
        for e, frame in enumerate(frames):
            if e % 4 == 3:
                frames_to_save.append(frame)
        for e, frame in enumerate(frames_to_save):
            Image.fromarray(frame[:,:,::-1]).resize((400, 300)).save(
                os.path.join(path_frames, str(e) + '.jpg'), quality=50)

def extract_features(path_in, net, num_layers_finetune, use_gpu, minimum_frames=45):

    # Create inference engine
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)

    # extract features
    for dataset in ["train", "valid"]:
        videos_dir = os.path.join(path_in, f"videos_{dataset}")
        features_dir = os.path.join(path_in, f"features_{dataset}_num_layers_to_finetune={num_layers_finetune}")
        video_files = glob.glob(os.path.join(videos_dir, "*", "*.mp4"))

        print(f"\nFound {len(video_files)} videos to process in the {dataset}set")

        for video_index, video_path in enumerate(video_files):
            print(f"\rExtract features from video {video_index + 1} / {len(video_files)}",
                  end="")
            path_out = video_path.replace(videos_dir, features_dir).replace(".mp4", ".npy")

            if os.path.isfile(path_out):
                print("\n\tSkipped - feature was already precomputed.")
            else:
                # Read all frames
                compute_features(video_path, path_out, inference_engine,
                                 minimum_frames=minimum_frames,  path_frames=None)

        print('\n')


def training_loops(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_state_dict = None
    best_top1 = 0.

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        new_lr = lr_schedule.get(epoch)
        if new_lr:
            print(f"update lr to {new_lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        net.train()
        train_loss, train_top1 = run_epoch(train_loader, net, criterion, optimizer, use_gpu)
        net.eval()
        valid_loss, valid_top1 = run_epoch(valid_loader, net, criterion, None, use_gpu)

        print('[%d] train loss: %.3f train top1: %.3f valid loss: %.3f top1: %.3f' % (epoch + 1, train_loss, train_top1,
                                                                                      valid_loss, valid_top1))

        if valid_top1 > best_top1:
            best_top1 = valid_top1
            best_state_dict = net.state_dict().copy()

    print('Finished Training')
    return best_state_dict


def training_loops_counting(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule):
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion_weights = convert_class_weight({'length': 31, 'mapping': {0: 0.2}})
    if use_gpu:
        criterion_weights = criterion_weights.cuda()
    criterion = nn.CrossEntropyLoss(
                        weight=criterion_weights)

    best_state_dict = None
    best_loss = 9999

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        new_lr = lr_schedule.get(epoch)
        if new_lr:
            print(f"update lr to {new_lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr


        net.train()
        train_loss, train_top1 = run_epoch_counting(train_loader, net, criterion, optimizer, use_gpu)

        net.eval()
        valid_loss, valid_top1 = run_epoch_counting(valid_loader, net, criterion, None, use_gpu)




        print('[%d] train loss: %.3f train top1: %.3f valid loss: %.3f top1: %.3f' % (epoch + 1, train_loss, train_top1,
                                                                                      valid_loss, valid_top1))

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = net.state_dict().copy()

    print('Finished Training')
    return best_state_dict


def run_epoch(data_loader, net, criterion, optimizer=None, use_gpu=False):
    running_loss = 0.0
    epoch_top_predictions = []
    epoch_labels = []

    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, targets]
        inputs, targets = data
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward + backward + optimize
        if net.training:
            # Run on each batch element independently
            outputs = [net(input_i) for input_i in inputs]
            # Concatenate outputs to get a tensor of size batch_size x num_classes
            outputs = torch.cat(outputs, dim=0)
        else:
            # Average predictions on the time dimension to get a tensor of size 1 x num_classes
            # This assumes validation operates with batch_size=1 and process all available features (no cropping)
            assert data_loader.batch_size == 1
            outputs = net(inputs[0])
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

    return loss, top1


def run_epoch_counting(data_loader, net, criterion, optimizer=None, use_gpu=False):
    running_loss = 0.0
    epoch_top_predictions = []
    epoch_labels = []

    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, targets]
        inputs, targets = data
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward + backward + optimize
        if net.training:
            # Run on each batch element independently
            outputs = [net(input_i) for input_i in inputs]
            # Concatenate outputs to get a tensor of size batch_size x num_classes
            outputs = torch.cat(outputs, dim=0)

            # take only targets one batch
            targets = targets[:,0]
            # realign the number of outputs
            min_pred_number = min(outputs.shape[0], targets.shape[0])
            targets = targets[0:min_pred_number]
            outputs = outputs[0:min_pred_number]
        else:
            # Average predictions on the time dimension to get a tensor of size 1 x num_classes
            # This assumes validation operates with batch_size=1 and process all available features (no cropping)
            assert data_loader.batch_size == 1
            outputs = net(inputs[0])
            # outputs = torch.mean(outputs, dim=0, keepdim=True)
            targets = targets[0]
            # realign the number of outputs
            min_pred_number = min(outputs.shape[0], targets.shape[0])
            targets = targets[0:min_pred_number]
            outputs = outputs[0:min_pred_number]

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

    return loss, top1


def convert_class_weight(class_weight):
    if class_weight is None:
        return None

    if isinstance(class_weight, dict):
        class_weight_ = class_weight['length'] * [1.]
        for index, weight in class_weight['mapping'].items():
            class_weight_[index] = weight
        class_weight = class_weight_
    return torch.Tensor(class_weight)