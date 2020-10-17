import queue
from threading import Thread

import numpy as np
import cv2 as cv2
import torch


class InferenceEngine(Thread):

    def __init__(self, net, use_gpu=False):
        Thread.__init__(self)
        self.net = net
        self.use_gpu = use_gpu
        if use_gpu:
            self.net.cuda()
        self._queue_in = queue.Queue(1)
        self._queue_out = queue.Queue(1)
        self._shutdown = False

    @property
    def expected_frame_size(self):
        return self.net.expected_frame_size

    @property
    def fps(self):
        return self.net.fps

    @property
    def step_size(self):
        return self.net.step_size

    def put_nowait(self, clip):
        if self._queue_in.full():
            # Remove one clip
            self._queue_in.get_nowait()
            print("*** Unused frames ***")
        self._queue_in.put_nowait(clip)

    def get_nowait(self):
        if self._queue_out.empty():
            return None
        return self._queue_out.get_nowait()

    def stop(self):
        self._shutdown = True

    def run(self):
        while not self._shutdown:
            try:
                clip = self._queue_in.get(timeout=1)
            except queue.Empty:
                clip = None

            if clip is not None:
                predictions = self.process_clip(clip)
                if self._queue_out.full():
                    # Remove one frame
                    self._queue_out.get_nowait()
                    print("*** Unused predictions ***")
                self._queue_out.put(predictions, block=False)

    def process_clip(self, clip, training=False):
        with torch.no_grad():
            clip = self.net.preprocess(clip)

            if self.use_gpu:
                clip = clip.cuda()

            predictions = self.net(clip)
        if training:
            if isinstance(predictions, list):
                predictions = [pred.cpu().numpy() for pred in predictions]
            else:
                predictions = predictions.cpu().numpy()
        else:
            if isinstance(predictions, list):
                predictions = [pred.cpu().numpy()[0] for pred in predictions]
            else:
                predictions = predictions.cpu().numpy()[0]
        return predictions

    def process_clip_features_map(self, clip, layer=-2):
        """
        extract features map of a clip at given layer
        :param clip:
        :param layer: layer position where to extract features map
        :return:
        features map
        """
        with torch.no_grad():
            clip = self.net.preprocess(clip)

            if self.use_gpu:
                clip = clip.cuda()

            predictions = self.net.cnn[0:layer](clip)
        if isinstance(predictions, list):
            predictions = [pred.cpu().numpy() for pred in predictions]
        else:
            predictions = predictions.cpu().numpy()
        return predictions


def run_inference_engine(inference_engine, framegrabber, post_processors, results_display, path_out):

    # Initialization of a few variables
    clip = np.random.randn(1, inference_engine.step_size, inference_engine.expected_frame_size[0],
                           inference_engine.expected_frame_size[1], 3)
    frame_index = 0
    display_error = None
    video_recorder = None
    video_recorder_raw = None

    # Start threads
    inference_engine.start()
    framegrabber.start()

    # Begin calorie estimation
    while True:
        frame_index += 1

        # Grab frame if possible
        img_tuple = framegrabber.get_image()
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

        # Update calories
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
    framegrabber.stop()
    inference_engine.stop()
    if video_recorder is not None:
        video_recorder.release()
    if video_recorder_raw is not None:
        video_recorder_raw.release()

    if display_error:
        raise display_error


def load_weights(checkpoint_path):
    try:
        return torch.load(checkpoint_path)
    except:
        raise Exception('ERROR - Weights file missing: {}. To download, please go to '
                        'https://20bn.com/licensing/sdk/evaluation and follow the '
                        'instructions.'.format(checkpoint_path))
