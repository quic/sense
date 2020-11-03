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

Create a new virtual environment. The following instruction uses [conda](https://docs.conda.io/en/latest/miniconda.html) (recommended).
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
After downloading the weights, be sure to unzip, find the `backbone` folder and move it 
into `20bn-realtimenet/resources`. In the end, your resources folder should look like this:

```
resources
    ├── backbone
        ├── strided_inflated_efficientnet.ckpt
        ├── strided_inflated_mobilenet.ckpt
    ├── fitness_activity_recognition
        ├── ...
    ├── gesture_detection
        ├── ...
    ├── ....
```

Note: only the `backbone` files are new, the other folders should not require any change.


## Available live demos
Inside the `20bn-realtimenet/scripts` directory, you will find 3 Python scripts,
`run_gesture_recognition.py`, `run_fitness_tracker.py`, and `run_calorie_estimation.py`.

### 1. Gesture Recognition

`scripts/gesture_recognition.py` applies our pre-trained models to hand gesture recognition.
30 gestures are supported (see full list 
[here](https://github.com/TwentyBN/20bn-realtimenet/blob/7651d24967de7eb12912297747de8174950eb74e/realtimenet/downstream_tasks/gesture_recognition/__init__.py)):

```shell
PYTHONPATH=./ python scripts/run_gesture_recognition.py
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
PYTHONPATH=./ python scripts/run_fitness_tracker.py --weight=65 --age=30 --height=170 --gender=female
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
PYTHONPATH=./ python scripts/run_calorie_estimation.py --weight=65 --age=30 --height=170 --gender=female
```

The estimated calorie estimates are roughly in the range produced by wearable devices, though they have not been verified in terms of accuracy. 
From our experiments, our estimates correlate well with the workout intensity (intense workouts burn more calories) so, regardless of the absolute accuracy, it should be fair to use this metric to compare one workout to another.


### Transfer learning: build your own demo

This repo implements scripts that can be used to finetune one of our pretrained models on your specific data and evaluate the obtained model. 

#### 1. Training

Run this command to train a customized classifier on top of one of our features extractor:
```shell
PYTHONPATH=./ python scripts/train_classifier.py --path_in=/path/to/your/dataset/ [--use_gpu] [--num_layers_to_finetune=9]
```

This script expects training videos to be organized according to this structure:

```
    /path/to/your/dataset/
    ├── videos_train/
        ├── label1/
            ├── video1.mp4
            ├── video2.mp4
            ├── ...
        ├── label2/
            ├── video3.mp4
            ├── video4.mp4
            ├── ...
        ├── ...
    ├── videos_valid/
        ├── label1/
            ├── video5.mp4
            ├── video6.mp4
            ├── ...
        ├── label2/
            ├── video7.mp4
            ├── video8.mp4
            ├── ...
        ├── ...
```
- Two top-level folders: one for the training data, one for the validation data.
- One sub-folder for each label with as many videos as you want (but at least one!)
- Requirement: videos should have a framerate of 16 fps or higher.

#### 2. Live demo

The training script should produce a checkpoint file called `classifier.checkpoint` at the root of the dataset folder.
You can now run it live using the following script:

```shell
PYTHONPATH=./ python scripts/run_custom_classifier.py --custom_classifier=/path/to/your/dataset/ [--use_gpu]
```

## Running on an iOS Device and CoreML Conversion

If you're interested in mobile app development and want to run our models on iOS devices, please check out [20bn-realtimenet-ios](https://github.com/TwentyBN/20bn-realtimenet-iOS) for step by step instructions on how to get our gesture demo to run on an iOS device.
One of the steps involves converting our Pytorch models to the CoreML format, which can be done from this repo using the following script:

```shell
python scripts/conversion/convert_to_coreml.py --backbone=efficientnet --classifier=efficient_net_gesture_control --output_name=realtimenet
```

## Citation

We now have a [blogpost](https://medium.com/twentybn/towards-situated-visual-ai-via-end-to-end-learning-on-video-clips-2832bd9d519f) you can cite:

```bibtex
@misc{realtimenet2020blogpost,
    author = {Guillaume Berger and Antoine Mercier and Florian Letsch and Cornelius Boehm and Sunny Panchal and Nahua Kang and Mark Todorovich and Ingo Bax and Roland Memisevic},
    title = {Towards situated visual AI via end-to-end learning on video clips},
    howpublished = {\url{https://medium.com/twentybn/towards-situated-visual-ai-via-end-to-end-learning-on-video-clips-2832bd9d519f}},
    note = {online; accessed 23 October 2020},
    year=2020,
}
```

## License 

The code is copyright (c) 2020 Twenty Billion Neurons GmbH under an MIT Licence. See the file LICENSE for details. Note that this license 
only covers the source code of this repo. Pretrained weights come with a separate license available [here](https://20bn.com/licensing/sdk/evaluation).

This repo uses PyTorch, which is licensed under a 3-clause BSD License. See the file LICENSE_PYTORCH for details.
