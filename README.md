# 20bn-realtimenet

This repo contains the inference code for two neural networks that were pretrained on millions of videos. Both neural
networks are rather small and should smoothly run in real time on a CPU. 

## Installation

The following steps have been confirmed to work on a Linux machine (Ubuntu 18.04 LTS). They probably also work on MacOS/Windows.

To begin, clone this repository to a local directory:
```
git clone git@github.com:TwentyBN/20bn-realtimenet.git
cd 20bn-realtimenet
```


### Dependencies

Create [conda](https://docs.conda.io/en/latest/miniconda.html) environment:

```shell
conda create -y -n realtimenet python=3.6
conda activate realtimenet
```


Install Python dependencies.

```shell
pip install -r requirements.txt
```

Note: `pip install -r requirements.txt` will install the CPU-only version of PyTorch. To run inference on your GPU, 
another version of PyTorch should be installed (e.g. `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`, 
see all available options [here](https://pytorch.org/)).


### Pretrained weights

Pretraineds weights can be downloaded from this [google drive](https://drive.google.com/drive/folders/11UFnZDcpqehMYpv88PSE4m3bIPLiAZXh?usp=sharing). 
Downloaded files should be placed into `20bn-realtimenet/resources`.

## Available scripts

### Calorie estimation


```shell
python scripts/calorie_estimation.py --weight=65 --age=30 --height=170 --gender=female
```

Weight, age, height should be respectively given in kilograms, years and centimeters. If not provided, default values will be used.

Some additional arguments can be used to grab frames from a different source:
```
  --camera_id=CAMERA_ID           ID of the camera to stream from
  --path_in=FILENAME              Video file to stream from. This assumes that the video was encoded at 16 fps.
```

It is also possible to save the display window to a video file using:
```
  --path_out=FILENAME             Video file to stream to
```

#### Ideal setup:

Best performance are obtained in these conditions: 
- Camera on the floor 
- Body fully visible (head-to-toe) 
- Clean background 

### Fitness Activity Tracking

```shell
python scripts/fitness_tracker.py --weight=65 --age=30 --height=170 --gender=female
```

![](gifs/fitness_tracking.gif "Test")

### Gesture Recognition

```shell
python scripts/gesture_recognition.py
```

![](gifs/gesture_recognition.gif "Test")
