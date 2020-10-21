# 20bn-realtimenet

`20bn-realtimenet` is an inference engine for two lightweight neural networks that were pre-trained on millions of videos,
including a gesture recognition model and a fitness activity tracking model.
Both neural networks are small, efficient, and run smoothly in real time on a CPU. 

## Getting Started
The following steps are confirmed to work on Linux (Ubuntu 18.04 LTS and 20.04 LTS) and macOS (Catalina 10.15.7).

### 1. Clone the Repository

To begin, clone this repository to a local directory of your choice:
```
git clone https://github.com/TwentyBN/20bn-realtimenet.git
cd 20bn-realtimenet
```

### 2. Install Dependencies

Create a new virtual environment. The following instruction uses [conda](https://docs.conda.io/en/latest/miniconda.html).
You can also create a new virtual environment with `virtualenv`.

```shell
conda create -y -n realtimenet python=3.6
conda activate realtimenet
```

Install Python dependencies:

```shell
pip install -r requirements.txt
```

Note: `pip install -r requirements.txt` only installs the CPU-only version of PyTorch.
To run inference on your GPU,  another version of PyTorch should be installed. For instance:
```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
``` 
See all available options [here](https://pytorch.org/).


### 3. Download Pre-trained Weights

Pre-trained weights can be downloaded from [here](https://20bn.com/licensing/sdk/evaluation).
Follow the link, register for your account, and you will be redirected to the download page.
After downloading the weights, be sure to unzip and place the contents of the directory into `20bn-realtimenet/resources`.

## Available scripts
Inside the `20bn-realtimenet/scripts` directory, you will find 3 Python scripts,
`gesture_recognition.py`, `fitness_tracker.py`, and `calorie_estimation.py`.

### 1. Gesture Recognition

`scripts/gesture_recognition.py` applies our pre-trained models to hand gesture recognition.
30 gestures are supported (see full list 
[here](https://github.com/TwentyBN/20bn-realtimenet/blob/7651d24967de7eb12912297747de8174950eb74e/realtimenet/downstream_tasks/gesture_recognition/__init__.py)):

```shell
PYTHONPATH=./ python scripts/gesture_recognition.py
```

![](gifs/gesture_recognition.gif)

*(full video can be found [here](https://drive.google.com/file/d/1G5OaCsPco_4H7F5-s6n2Mm3wI5V9K6WE/view?usp=sharing))*


### 2. Fitness Activity Tracking

`scripts/fitness_tracker.py` applies our pre-trained models to real-time fitness activity recognition and calorie estimation. 
In total, 80 different fitness exercises are recognized (see full list 
[here](https://github.com/TwentyBN/20bn-realtimenet/blob/d539046fe71e43e37ad439d08e093ea1f489bd29/realtimenet/downstream_tasks/fitness_activity_recognition/__init__.py)).

![](gifs/fitness_tracking.gif)

*(full video can be found [here](https://drive.google.com/file/d/1f1y0wg7Y1kpSBwKSEFx1TDoD5lGA8DtQ/view?usp=sharing))*

Usage:

```shell
PYTHONPATH=./ python scripts/fitness_tracker.py --weight=65 --age=30 --height=170 --gender=female
```

Weight, age, height should be respectively given in kilograms, years and centimeters. If not provided, default values will be used.

Some additional arguments can be used to change the streaming source:
```
  --camera_id=CAMERA_ID           ID of the camera to stream from
  --path_in=FILENAME              Video file to stream from. This assumes that the video was encoded at 16 fps.
```

It is also possible to save the display window to a video file using:
```
  --path_out=FILENAME             Video file to stream to
```

#### Ideal Setup:

For the best performance, the following is recommended: 
- Camera on the floor 
- Body fully visible (head-to-toe) 
- Clean background 


### 3. Calorie Estimation

In order to estimate burned calories, we trained a neural net to convert activity features to the corresponding [MET value](https://en.wikipedia.org/wiki/Metabolic_equivalent_of_task).
We then post-process these MET values (see correction and aggregation steps performed [here](https://github.com/TwentyBN/20bn-realtimenet/blob/7651d24967de7eb12912297747de8174950eb74e/realtimenet/downstream_tasks/calorie_estimation/calorie_accumulator.py)) 
and convert them to calories using the user's weight.

If you're only interested in the calorie estimation part, you might want to use `scripts/calorie_estimation.py` which has a slightly more
detailed display (see video [here](https://drive.google.com/file/d/1VIAnFPm9JJAbxTMchTazUE3cRRgql6Z6/view?usp=sharing) which compares two videos produced by that script).

Usage:
```shell
PYTHONPATH=./ python scripts/calorie_estimation.py --weight=65 --age=30 --height=170 --gender=female
```

The estimated calorie estimates are roughly in the range produced by wearable devices, though they have not been verified in terms of accuracy. 
From our experiments, our estimates correlate well with the workout intensity (intense workouts burn more calories) so, regardless of the absolute accuracy, it should be fair to use this metric to compare one workout to another.


## License 

The code is copyright (c) 2020 Twenty Billion Neurons GmbH under an MIT Licence. See the file LICENSE for details. Note that this license 
only covers the source code of this repo. Pretrained weights come with a separate license available [here](https://20bn.com/licensing/sdk/evaluation).

This repo uses PyTorch, which is licensed under a 3-clause BSD License. See the file LICENSE_PYTORCH for details.
