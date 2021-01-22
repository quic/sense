This page will explain how you can use the additional tools we provide to help you with 
your project. 

---

## Temporal Annotations Tool

Training a classifier can be improved using temporal annotations; adding tags to time-stamped 
video frames prior to training the neural network.

The dataset manager tool provides support for adding temporal annotations to your own dataset.

#### Step 0: Set up your project

If you haven't done so yet, run the `dataset_manager.py` script, open http://127.0.0.1:5000/ in
your browser and create a new project.

```commandline
PYTHONPATH=./ python tools/dataset_manager/dataset_manager.py
```

On the setup page you can also assign custom names to the temporal tags that will later be used
for annotating frames.

Once the project is created, fill the automatically created folders with your videos.

#### Step 1: Prepare the videos for annotation

Run the dataset manager tool as described above and navigate into your project.
There you will find a button to "Prepare Annotations", which will run the following preprocessing
steps:
1. Extract single frames from your videos 
2. Precompute features on those frames

The following additional folders should be added to your project as a result:
```
/path/to/your/dataset/
├── features_train
├── features_valid
├── frames_train
└── frames_valid
```

`features_<train|valid>/` will contain a `.npy` file for each video representing the extracted 
features (output of the frozen model layers when the video is run through the network).

`frames_<train|valid>/` will contain a list of images, which are the frames extracted for each video
and that will later be annotated.

Processing all of your videos can take a few minutes, depending on the size of your dataset.
Alternatively, the videos from a single class and train/valid split will be individually processed
when you click the "Annotate" button.

#### Step 2 : Annotating the videos

Now you can annotate the videos for the relevant classes by clicking the corresponding "Annotate"
button.
The interface will show the extracted frames from your video and you can select the appriate tag
for each one.
Per default, all frames are labeled "Background".

Once you're done tagging, submit the annotations and proceed to the next video.

**NOTE:** The model will perform best on annotations that occur after the first 2-3 seconds of the video

You can also quickly train a logistic regression model on all of the tags you have submitted so far,
which can serve as a rough guideline for further tagging and provides an impression of how well the
model can already identify your tags.
