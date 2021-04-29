SenseStudio is our one-stop, web-based platform to quickly get you started with
developing your own classifier. On this page you will find some tips and more 
detailed instructions on making the most of the tools we provide.


## Setting up your project

SenseStudio can be launched by running the `sense_studio.py` script and opening `http://127.0.0.1:5000/`
in your local browser.

```commandline
PYTHONPATH=./ python tools/sense_studio/sense_studio.py
```

On the welcome page you have the option of creating a new project by providing a project name and 
path, or, importing an existing project (previously created with SenseStudio).

Projects are initialised with the following structure:
```
/path/to/your/dataset/
├── videos_train
├── videos_valid
└── project_config.json
```

The project config stores project details and settings such as time created, classes, video
recording parameters, etc.


## Creating New Classes

New classes can be added from the `Project Details` tab. Creating a
new class automatically updates the video folder structure and project config. 

**NOTE**: Class names can be edited directly on SenseStudio from the `Project Details` page.


## Video Recording

Videos for each class can be split into training and validation sets and moved into their 
corresponding folders within the project. If you need to create new videos, you can do this
easily using the built-in video recorder on SenseStudio. Videos can be recorded by pressing the `Record`
button on the class card on the `Project Details` page, or, by clicking on the class from the dropdown
menu from the `Video Recording` tab. 

On the video recording page you can then set the desired pre-recording countdown and recording duration. 
We recommend setting the duration that best fits the intended action being performed -- it's okay to have 
some buffer (1-2 seconds) around the action and you'll notice the model still works quite well. For more 
advanced and precise classification, you can temporally annotate the videos instead to further distinguish 
the action from the background.

**Note**: Remember to allow permissions to access the camera when requested.


## Temporal Annotations

Training a classifier can be improved using temporal annotations (adding tags to time-stamped 
video frames prior to training the neural network).

Temporal annotations can be turned on in the settings window in the top-right corner
of the `Project Details` page. Once enabled, each class card will expand, and you will see a list of
tags, and a summary of the number of available videos that have been annotated. For now, 2 tags 
can be defined in addition to a default background class. Names can be edited by clicking on the edit icon 
on the card. 

When ready to annotate your videos, click on the "Annotate" button. On first click the videos in the
selected folder will be pre-processed to split the file into individual frames and set up the 
correct folder structure. This may take some time depending on how many videos you have.

You can then click on the video you want to annotate and be presented with a grid-view of each 
frame where the appropriate tag can be selected. Remember to click `Submit` at the bottom of the 
page when you are finished to return to the previous page.

**NOTE:** The model may not perform well on annotations within the first 2-3 seconds of the video and so
it is recommended that you collect videos with a small buffer at the beginning.


## Training

Once the dataset has been prepared, you can train your new custom classifier directly from SenseStudio
from the `Training` tab. 


## Testing

To evaluate your model, you can run live inference on your webcam stream or provide a local video
for the model to perform predictions on. Both options can be found on the "Testing" page.

