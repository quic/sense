import os
import glob

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from realtimenet import camera
from realtimenet import engine


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


def generate_data_loader(features_dir, classes, class2int,
                         num_timesteps=5, batch_size=16, shuffle=True):

    # Find pre-computed features and derive corresponding labels
    features = []
    labels = []
    for label in classes:
        feature_temp = glob.glob(f'{features_dir}/{label}/*.npy')
        features += feature_temp
        labels += [class2int[label]] * len(feature_temp)

    # Build dataloader
    dataset = FeaturesDataset(features, labels, num_timesteps=num_timesteps)
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


def extract_features(path_in, net, num_layer_finetune, use_gpu, minimum_frames=45):

    # Create inference engine
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)

    # extract features
    for dataset in ["train", "valid"]:
        videos_dir = os.path.join(path_in, f"videos_{dataset}")
        features_dir = os.path.join(path_in, f"features_{dataset}_{num_layer_finetune}")
        video_files = glob.glob(os.path.join(videos_dir, "*", "*.mp4"))

        print(f"\nFound {len(video_files)} videos to process")

        for video_index, video_path in enumerate(video_files):
            print(f"\rExtract features from video {video_index} / {len(video_files)}",
                  end="")
            path_out = video_path.replace(videos_dir, features_dir).replace(".mp4", ".npy")

            if os.path.isfile(path_out):
                print("\n\tSkipped - feature was already precomputed.")
            else:
                # Read all frames
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
                frames = uniform_frame_sample(np.array(frames), inference_engine.fps/video_fps)
                clip = np.array([frames]).astype(np.float32)

                # Inference
                if clip.shape[1] > minimum_frames:
                    predictions = inference_engine.infer(clip)
                    features = np.array(predictions)
                    os.makedirs(os.path.dirname(path_out), exist_ok=True)
                    np.save(path_out, features)

                else:
                    print(f"Video too short: {video_path}")


def training_loops(net, train_loader, valid_loader, use_gpu, num_epochs, lr_schedule):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

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

    print('Finished Training')


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
