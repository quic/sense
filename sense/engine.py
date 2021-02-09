import numpy as np
import queue
import torch

from threading import Thread
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from sense.downstream_tasks.nn_utils import RealtimeNeuralNet


class InferenceEngine(Thread):
    """
    InferenceEngine takes in a neural network and uses it to output predictions
    either using the local machine's CPU or GPU.
    """

    def __init__(self, net: RealtimeNeuralNet, use_gpu: bool = False):
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
    def fps(self) -> int:
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

    def infer(self, clip: np.ndarray, batch_size=None) -> Union[np.ndarray, List[np.ndarray]]:
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
        :param batch_size:
            Batch size to perform inference. Warning, only use if you did not remove
            padding from model.

        :return:
            Predictions from the neural network.
        """
        predictions = []
        with torch.no_grad():
            clip = self.net.preprocess(clip)

            if self.use_gpu:
                clip = clip.cuda()
            if batch_size is None:
                predictions = self.net(clip)
            else:
                for sub_clip in torch.Tensor.split(clip, batch_size):
                    if sub_clip.shape[0] >= self.net.num_required_frames_per_layer_padding[0]:
                        predictions.append(self.net(sub_clip))
                if isinstance(predictions[0], list):
                    predictions = list(zip(predictions))
                    predictions = [torch.cat(x, dim=0) for x in predictions]
                else:
                    predictions = torch.cat(predictions, dim=0)

        if isinstance(predictions, list):
            predictions = [pred.cpu().numpy() for pred in predictions]
        else:
            predictions = predictions.cpu().numpy()

        return predictions
