from typing import List
from typing import Union

from realtimenet.camera import VideoSource
from realtimenet.camera import VideoStream
from realtimenet.display import DisplayResults
from realtimenet.engine import InferenceEngine
from realtimenet.downstream_tasks.postprocess import PostProcessor

import cv2
import numpy as np
import torch.nn as nn


class Controller:

    def __init__(
            self,
            neural_network: nn.Module,
            post_processors: Union[PostProcessor, List[PostProcessor]],
            results_display: DisplayResults,
            camera_id: int,
            use_gpu: bool = True,
            path_out: str = None):
        # Initialize attributes

        self.net = neural_network
        self.inference_engine = None
        self.video_stream = None
        self.postprocessors = post_processors
        self.results_display = results_display
        self.path_out = path_out
        self.video_recorder = None
        self.video_recorder_raw = None
        self.display_error = None

        self.inference_engine = InferenceEngine(self.net, use_gpu=use_gpu)

        video_source = VideoSource(
            camera_id=camera_id,
            size=self.inference_engine.expected_frame_size
        )
        self.video_stream = VideoStream(video_source, self.inference_engine.fps)

    def display_predictions(self,
                            img: np.ndarray,
                            prediction: np.ndarray,
                            post_processed_data: dict):
        display_data = {'prediction': prediction, **post_processed_data}
        # Live display
        img_with_ui = self.results_display.show(img, display_data)

        # Recording
        if self.path_out:
            self.video_recorder = cv2.VideoWriter(self.path_out, 0x7634706d,
                                                  self.inference_engine.fps,
                                                  (img_with_ui.shape[1], img_with_ui.shape[0]))
            self.video_recorder_raw = cv2.VideoWriter(self.path_out.replace('.mp4', '_raw.mp4'),
                                                      0x7634706d,
                                                      self.inference_engine.fps,
                                                      (img.shape[1], img.shape[0]))

            self.video_recorder.write(img_with_ui)
            self.video_recorder_raw.write(img)

    def run_inference(self):
        clip = np.random.randn(1, self.inference_engine.step_size, self.inference_engine.expected_frame_size[0],
                               self.inference_engine.expected_frame_size[1], 3)

        frame_index = 0

        self._start_inference()
        while True:
            try:
                frame_index += 1

                # Grab frame if possible
                img_tuple = self.video_stream.get_image()
                # If not possible, stop
                if img_tuple is None:
                    break

                # Unpack
                img, numpy_img = img_tuple

                clip = np.roll(clip, -1, 1)
                clip[:, -1, :, :, :] = numpy_img

                if frame_index == self.inference_engine.step_size:
                    # A new clip is ready
                    self.inference_engine.put_nowait(clip)

                frame_index = frame_index % self.inference_engine.step_size

                # Get predictions
                prediction = self.inference_engine.get_nowait()

                post_processed_data = {}
                for post_processor in self.postprocessors:
                    post_processed_data.update(post_processor(prediction))

                # self._apply_commands(post_processed_data)
                self.display_predictions(
                    img=img,
                    prediction=prediction,
                    post_processed_data=post_processed_data
                )

            except Exception as exception:
                self.display_error = exception
                break

        self._stop_inference()

    def _start_inference(self):
        print("Starting inference")
        self.inference_engine.start()
        self.video_stream.start()

    def _stop_inference(self):
        print("Stopping inference")
        cv2.destroyAllWindows()
        self.video_stream.stop()
        self.inference_engine.stop()

        if self.video_recorder is not None:
            self.video_recorder.release()

        if self.video_recorder_raw is not None:
            self.video_recorder_raw.release()

        if self.display_error:
            raise self.display_error
