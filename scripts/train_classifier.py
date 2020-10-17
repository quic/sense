#!/usr/bin/env python
"""
Real time detection of 30 hand gestures.

Usage:
  train_classifier.py    --path_in=PATH
                         [--num_layer_finetune=NUM]
                         [--use_gpu]
  train_classifier.py (-h | --help)

Options:
  --path_in=PATH              path to the dataset folder following the structure described in the readme
  --num_layer_finetune=NUM    Number layer to finetune [default: 2]

"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from docopt import docopt
import os
import glob
import numpy as np
from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet import camera
from realtimenet import engine
from realtimenet import feature_extractors
import json



if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    path_in = args['--path_in']
    use_gpu = args['--use_gpu']
    num_layer_finetune = int(args['--num_layer_finetune'])



    # Load feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    checkpoint = torch.load('resources/strided_inflated_efficientnet.ckpt')
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    # Concatenate feature extractor and met converter
    net = feature_extractor

    # Create inference engine, video streaming and display instances
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)


    postprocessor = []
    display_ops = []


    # list the videos files
    videos_dir = os.path.join(path_in, "videos_train")
    features_dir = os.path.join(path_in, "features_train")
    classes = os.listdir(videos_dir)
    classes = [x for x in classes if not x.startswith('.')]
    print(classes)


    # extract features
    for dataset in ["train", "valid"]:
        videos_dir = os.path.join(path_in, "videos_" + dataset)
        features_dir = os.path.join(path_in, "features_" + dataset+ "_" + str(num_layer_finetune))
        number_videos_found = len(glob.glob(os.path.join(videos_dir, "*", "*.mp4")))
        number_videos_processed = 0
        print(f"Found {number_videos_found} videos to process")

        for label in classes:
            videos = os.listdir(os.path.join(videos_dir, label))
            videos = [x for x in videos if not x.startswith('.')]
            for video in videos:
                number_videos_processed += 1
                print(f"extract features from video {number_videos_processed} / {number_videos_found}")
                path_out = os.path.join(features_dir, label, video.replace(".mp4", ".npy"))
                if os.path.isfile(path_out):
                    print("features found for this file, skip")
                else:
                    # reset the buffer
                    net.train()
                    net.eval()

                    file_path = os.path.join(videos_dir, label, video)
                    path_out = os.path.join(features_dir, label, video.replace(".mp4", ".npy"))

                    video_source = camera.VideoSource(camera_id=None,
                                                      size=inference_engine.expected_frame_size,
                                                      filename=file_path)
                    frames = []
                    features = []
                    while True:
                        images = video_source.get_image()
                        if images is None:
                            break
                        else:
                            image, image_rescaled = images
                            frames.append(image_rescaled)
                            if len(frames) == net.step_size:
                                clip = np.array([frames]).astype(np.float32)
                                frames = []
                                if num_layer_finetune > 0:
                                    predictions = inference_engine.process_clip_features_map(clip,
                                                                                         layer=-num_layer_finetune)
                                else:
                                    predictions = inference_engine.process_clip(clip)
                                features.append(predictions)
                    if len(features) < 13:
                        print("video too short")
                    else:
                        features = np.array(features[12:])
                        os.makedirs(os.path.dirname((path_out)), exist_ok=True)
                        np.save(path_out, features)



    # finetune the model

    y_train, y_valid = [], []
    X_train, X_valid = [], []
    class2int = {x:e for e,x in enumerate(classes)}


    def generate_data_loader(features_dir, classes, shuffle=True):
        X = []
        y = []
        for label in classes:
            features = os.listdir(os.path.join(features_dir, label))
            # used to remove .DSstore files on mac
            features = [x for x in features if not x.startswith('.')]
            for feature in features:
                feature = np.load(os.path.join(features_dir, label, feature))
                X += list(feature)
                y += [class2int[label]] * feature.shape[0]
        X = np.array(X)
        y = np.array(y)
        data = []
        for i in range(len(X)):
            data.append([X[i], y[i]])
        data_loader = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=64)
        return data_loader

    trainloader = generate_data_loader(os.path.join(path_in, "features_train_" + str(num_layer_finetune)), classes)
    validloader = generate_data_loader(os.path.join(path_in, "features_valid_" + str(num_layer_finetune)), classes)

    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=len(classes))
    if num_layer_finetune > 0:
        net.cnn = net.cnn[-num_layer_finetune:]
        net = Pipe(feature_extractor, gesture_classifier)
    else:
        net = gesture_classifier
    net.train()
    if use_gpu:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        net.train()
        top_pred = []
        label = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            label += list(labels.cpu().numpy())
            top_pred += list(outputs.argmax(dim=1).cpu().numpy())

            # print statistics
            running_loss += loss.item()
        percentage = np.sum((np.array(label) == np.array(top_pred))) / len(top_pred)
        train_top1 = percentage
        train_loss =  running_loss / len(trainloader)

        running_loss = 0.0
        top_pred = []
        label = []
        net.eval()
        for i, data in enumerate(validloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()


                # forward + backward + optimize
                outputs = net(inputs)
                label += list(labels.cpu().numpy())
                top_pred += list(outputs.argmax(dim=1).cpu().numpy())

                # print statistics
                running_loss += loss.item()
        percentage = np.sum((np.array(label) == np.array(top_pred))) / len(top_pred)
        valid_top1 = percentage
        valid_loss =  running_loss / len(trainloader)

        print('[%d] train loss: %.3f train top1: %.3f valid loss: %.3f top1: %.3f' % (epoch + 1, train_loss, train_top1,
                                                                                      valid_loss, valid_top1))

    print('Finished Training')
    print("score on videos")
    features_dir = os.path.join(path_in, "features_valid_" + str(num_layer_finetune))

    pred = []
    y = []
    for label in classes:
        features = os.listdir(os.path.join(features_dir, label))
        # used to remove .DSstore files on mac
        features = [x for x in features if not x.startswith('.')]
        for feature in features:
            feature = np.load(os.path.join(features_dir, label, feature))
            feature = torch.Tensor(feature).cuda()
            with torch.no_grad():
                output = net(feature).cpu().numpy()
            top_pred = output.mean(axis=0)
            pred.append(top_pred.argmax())
            y.append(class2int[label])
    pred = np.array(pred)
    y = np.array(y)
    percent = (np.sum(pred == y) / len(pred))
    print(f"top 1 : {percent}")
    if num_layer_finetune > 0:
        state_dict = {**net.feature_extractor.state_dict(), **net.feature_converter.state_dict()}
    else:
        state_dict = net.state_dict()
    torch.save(state_dict, os.path.join(path_in, "classifier.checkpoint"))
    json.dump(class2int, open(os.path.join(path_in,"class2int.json"), "w"))


