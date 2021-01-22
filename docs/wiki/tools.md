This page will explain how you can use the additional tools we provide to help you with 
your project. 

---

## Temporal Annotations Tool

Training a classifier can be improved using temporal annotations; adding labels to time-stamped 
video frames prior to training the neural network. We provide two scripts for this task:
- `prepare_annotation.py`: Pre-processing script to help prepare frames for the annotation task
- `anotation.py`: Script for frame-wise annotation

#### Step 0: Prepare your dataset folder
Before you begin, ensure that you've followed the steps [here](https://github.com/TwentyBN/sense/tree/master#build-your-own-classifier) 
to correctly structure your dataset.

#### Step 1 : Prepare the videos for annotation

After the dataset folder is ready, run the `prepare_annotation.py` script:

```commandline
PYTHONPATH=./ python tools/annotation/prepare_annotation.py --data_path=/path/to/your/dataset/
```

This step will divide the videos inside each class-folder for both videos_train and videos_valid and  
extract features for each of them. You will see the progress in the terminal window.

Your dataset directory should then have the following structure:
```
/path/to/your/dataset/
├── features_train
├── features_valid
├── frames_train
├── frames_valid
├── videos_train
└── videos_valid
```

`features_<train|valid>/` will contain a `.npy` file for each video representing the extracted 
features (output of the frozen model layers when the video is run through the network).

`frames_<train|valid>/` will contain a list of images, which are the frames extracted for each video
and that will be later annotated.

#### Step 2 : Annotating the videos

Next, run the `annotation.py` script:

```commandline
PYTHONPATH=./ python tools/annotation/annotation.py --data_path=/path/to/your/dataset/ --split=[train or valid] --label=[Label_of_videos_to_annotate]
```

This command will launch a web server which provides an interface for you to annotate each video by 
frame. After selecting from the list of available videos, you will be taken to a page with the 
frames presented in a grid with radio button to select the correct label.

**NOTE:** The model will perform best on annotations that occur after the first 2-3 seconds of the video
