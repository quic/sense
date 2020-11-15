import queue
from threading import Thread
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from realtimenet.display import DisplayResults
from realtimenet.camera import VideoStream
from realtimenet.downstream_tasks.postprocess import PostProcessor

import cv2 as cv2
import numpy as np
import torch
import torch.nn as nn


class InferenceEngine(Thread):
    """
    InferenceEngine takes in a neural network and uses it to output predictions
    either using the local machine's CPU or GPU.
    """
    def __init__(self, net: nn.Module, use_gpu: bool = False):
        """
        :param net:
            The neural network to be run by the inference engine.
        :param use_gpu:
            Whether to leverage CUDA or not for neural network inference.
        """
        Thread.__init__(self)
        self.net = net
        self.use_gpu = use_gpu
        if use_gpu:
            self.net.cuda()
        self._queue_in = queue.Queue(1)
        self._queue_out = queue.Queue(1)
        self._shutdown = False

    @property
    def expected_frame_size(self) -> Tuple[int, int]:
        """Return the frame size of the video source input."""
        return self.net.expected_frame_size

    @property
    def fps(self) -> float:
        """Frame rate of the inference engine's neural network."""
        return self.net.fps

    @property
    def step_size(self) -> int:
        """The step size of the inference engine's neural network."""
        return self.net.step_size

    def put_nowait(self, clip: np.ndarray):
        """
        Add a new clip to the input queue of inference engine for prediction.

        :param clip:
            The video frame to be added to the inference engine's input queue.
        """
        if self._queue_in.full():
            # Remove one clip
            self._queue_in.get_nowait()
        self._queue_in.put_nowait(clip)

    def get_nowait(self) -> Optional[np.ndarray]:
        """
        Return a clip from the output queue of the inference engine if available.
        """
        if self._queue_out.empty():
            return None
        return self._queue_out.get_nowait()

    def stop(self):
        """Terminate the inference engine."""
        self._shutdown = True

    def run(self):
        """
        Keep the inference engine running and inferring predictions from input video frames.
        """
        while not self._shutdown:
            try:
                clip = self._queue_in.get(timeout=1)
            except queue.Empty:
                clip = None

            if clip is not None:
                predictions = self.infer(clip)

                # Remove time dimension
                if isinstance(predictions, list):
                    predictions = [pred[0] for pred in predictions]
                else:
                    predictions = predictions[0]

                if self._queue_out.full():
                    # Remove one frame
                    self._queue_out.get_nowait()
                    print("*** Unused predictions ***")
                self._queue_out.put(predictions, block=False)

    def infer(self, clip: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Infer and return predictions given the input clip from video source.
        Note that the output is either a numpy.ndarray type or a list consisting
        of numpy.ndarray.

        For an inference engine that runs a neural network, which producing a single output,
        the returned object is a numpy.ndarray of shape (T, C). `T` represents
        the number of time steps and  is dependent on the length of the provided input clip.
        `C` represents the number of output channels while

        For an inference engine running a multi-output neural network, the returned object
        is a list of numpy.ndarray, one for each output.

        :param clip:
            The video frame to be inferred.

        :return:
            Predictions from the neural network.
        """
        with torch.no_grad():
            clip = self.net.preprocess(clip)

            if self.use_gpu:
                clip = clip.cuda()

            predictions = self.net(clip)

        if isinstance(predictions, list):
            predictions = [pred.cpu().numpy() for pred in predictions]
        else:
            predictions = predictions.cpu().numpy()

        return predictions


def run_inference_engine(
        inference_engine: InferenceEngine,
        video_stream: VideoStream,
        post_processors: List[PostProcessor],
        results_display: DisplayResults,
        path_out: Optional[str]):
    """
    Start the video stream and the inference engine, process and display
    the prediction from the neural network.

    :param inference_engine:
        An instance of InferenceEngine for neural network inferencing.
    :param video_stream:
        An instance of VideoStream to feed video frames to InferenceEngine.
    :param post_processors:
        A list of subclasses of PostProcessor for post-processing neural network outputs.
    :param results_display:
        An instance of DisplayResults to display neural network predictions on the screen.
    :param path_out:
        Path to store the recorded video that includes predictions from the inference engine.
    """

    # Initialization of a few variables
    clip = np.random.randn(1, inference_engine.step_size, inference_engine.expected_frame_size[0],
                           inference_engine.expected_frame_size[1], 3)
    frame_index = 0
    display_error = None
    video_recorder = None
    video_recorder_raw = None

    # Start threads
    inference_engine.start()
    video_stream.start()

    # Begin inferencing
    while True:
        frame_index += 1

        # Grab frame if possible
        img_tuple = video_stream.get_image()
        # If not possible, stop
        if img_tuple is None:
            break

        # Unpack
        img, numpy_img = img_tuple

        clip = np.roll(clip, -1, 1)
        clip[:, -1, :, :, :] = numpy_img

        if frame_index == inference_engine.step_size:
            # A new clip is ready
            inference_engine.put_nowait(clip)

        frame_index = frame_index % inference_engine.step_size

        # Get predictions
        prediction = inference_engine.get_nowait()

        # Post process predictions and display the output
        post_processed_data = {}
        for post_processor in post_processors:
             post_processed_data.update(post_processor(prediction))

        try:
            display_data = {'prediction': prediction, **post_processed_data}
            # Live display
            img_with_ui = results_display.show(img, display_data)

            # Recording
            if path_out:
                if video_recorder is None or video_recorder_raw is None:
                    video_recorder = cv2.VideoWriter(path_out, 0x7634706d, inference_engine.fps,
                                                     (img_with_ui.shape[1], img_with_ui.shape[0]))
                    video_recorder_raw = cv2.VideoWriter(path_out.replace('.mp4', '_raw.mp4'), 0x7634706d,
                                                         inference_engine.fps, (img.shape[1], img.shape[0]))

                video_recorder.write(img_with_ui)
                video_recorder_raw.write(img)

        except Exception as exception:
            display_error = exception
            break

        # Press escape to exit
        if cv2.waitKey(1) == 27:
            break

    # Clean up
    cv2.destroyAllWindows()
    video_stream.stop()
    inference_engine.stop()
    if video_recorder is not None:
        video_recorder.release()
    if video_recorder_raw is not None:
        video_recorder_raw.release()

    if display_error:
        raise display_error


def load_weights(checkpoint_path: str):
    """
    Load weights from a checkpoint file.

    :param checkpoint_path:
        A string representing the absolute/relative path to the checkpoint file.
    """
    try:
        return torch.load(checkpoint_path, map_location='cpu')
    except:
        raise Exception('ERROR - Weights file missing: {}. To download, please go to '
                        'https://20bn.com/licensing/sdk/evaluation and follow the '
                        'instructions.'.format(checkpoint_path))
