import cv2
import numpy as np
import queue
import time

from threading import Thread
from typing import Optional
from typing import Tuple


def uniform_frame_sample(video, sample_rate):
    """
    Uniformly sample video frames according to the provided sample_rate.
    """
    available_frames = video.shape[0]
    required_frames = np.round(sample_rate * available_frames).astype(np.int32)

    # Get evenly spaced indices. When upsampling, include both endpoints.
    new_indices = np.linspace(0, available_frames - 1, num=required_frames, endpoint=sample_rate >= 1.)

    # Center the indices
    offset = ((available_frames - 1) - new_indices[-1]) / 2
    new_indices += offset

    # Round to closest integers
    new_indices = new_indices.round().astype(np.int32)
    return video[new_indices]


class VideoSource:
    """
    VideoSource captures frames from a camera or a video source file.
    """

    def __init__(self,
                 filename: str = None,
                 size: Tuple[int, int] = None,
                 camera_id: int = 0,
                 preserve_aspect_ratio: bool = True,
                 target_fps: Optional[int] = None):
        """
        :param filename:
            Path to a video file.
        :param size:
            The expected frame size of the video.
        :param camera_id:
            Device index for the camera.
        :param preserve_aspect_ratio:
            Whether to preserve the aspect ratio of the video frames.
        :param target_fps:
            Framerate that the video should be sampled to. If None is given, framerate of the video is left unchanged.
            Only relevant if the video is read from a file.
        """
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self._frames = None
        self._frame_idx = 0

        if filename:
            self._cam = cv2.VideoCapture(filename)

            if target_fps is not None:
                self._frames = self._read_and_resample_frames(target_fps)
        else:
            self._cam = cv2.VideoCapture(camera_id)
            self._cam.set(3, 480)  # Set frame width to 480
            self._cam.set(4, 640)  # Set frame height to 640

    def _read_and_resample_frames(self, target_fps):
        video_fps = self._cam.get(cv2.CAP_PROP_FPS)

        video = []
        ret, frame = self._cam.read()
        while ret:
            video.append(frame)
            ret, frame = self._cam.read()

        return uniform_frame_sample(np.array(video), target_fps / video_fps)

    def _get_frame(self):
        if self._frames is not None:
            if self._frame_idx < len(self._frames):
                frame = self._frames[self._frame_idx]
                self._frame_idx += 1
                return frame
        else:
            ret, frame = self._cam.read()
            if ret:
                return frame

        return None

    def get_image(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Capture image from video stream frame-by-frame.
        The captured image and a scaled copy of the image are returned.
        """
        img = self._get_frame()
        if img is not None:
            img_copy = img.copy()
            if self.preserve_aspect_ratio:
                img_copy = self.pad_to_square(img)
            scaled_img = cv2.resize(img_copy, self.size) if self.size else img
            return img, scaled_img
        else:
            # Could not grab another frame (file ended?)
            return None

    def pad_to_square(self, img):
        """Pad an image to the shape of a square with borders."""
        square_size = max(img.shape[0:2])
        pad_top = int((square_size - img.shape[0]) / 2)
        pad_bottom = square_size - img.shape[0] - pad_top
        pad_left = int((square_size - img.shape[1]) / 2)
        pad_right = square_size - img.shape[1] - pad_left
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)


class VideoStream(Thread):
    """
    Thread that reads frames from the video source at a given frame rate.
    """

    def __init__(self, video_source: VideoSource, fps: float, queue_size: int = 4):
        """
        :param video_source:
            An instance of VideoSource that represents a camera stream or video file.
        :param fps:
            Frame rate of the inference engine.
        :param queue_size:
            Size of the FIFO queue that stores a tuple of image and scaled image.
        """
        Thread.__init__(self)
        self.video_source = video_source
        self.frames = queue.Queue(queue_size)
        self.fps = fps
        self.delta_t = 1.0 / self.fps
        self._shutdown = False

    def stop(self):
        """Stop the VideoStream instance."""
        self._shutdown = True

    def get_image(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get an image frame from the FIFO queue of frames."""
        return self.frames.get()

    def run(self):
        while not self._shutdown:
            time_start = time.perf_counter()
            image_tuple = self.video_source.get_image()

            if self.frames.full():
                # Remove one frame
                self.frames.get_nowait()
                print("*** Frame skipped ***")

            self.frames.put(image_tuple, False)

            # Last frame was a None
            if image_tuple is None:
                self.stop()
                continue

            # wait before the next framegrab to enforce a certain FPS
            elapsed = time.perf_counter() - time_start
            delay = self.delta_t - elapsed
            if delay > 0:
                time.sleep(delay)


class VideoWriter:
    """
    VideoWriter writes a video file.
    """
    def __init__(self, path: str, fps: float, resolution):
        """
        :param path:
            Path to the video file to be written.
        :param fps:
            The number of frames per second.
        :param resolution:
            Frame size for the video.
        """
        self.writer = cv2.VideoWriter(path, 0x7634706d, fps, resolution)
        self.delta_t = 1.0 / fps
        self._last_time = -1
        self.last_time_written = None

    def write(self, frame):
        """Write an image frame to the video."""
        now = time.perf_counter()
        if now - self._last_time >= self.delta_t:
            self.last_time_written = now
            self.writer.write(frame)

    def release(self):  # noqa: D102
        self.writer.release()
