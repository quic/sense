import cv2
import numpy as np
import queue
import time

from threading import Thread
from typing import Optional
from typing import Tuple


class VideoSource:
    """
    VideoSource captures frames from a camera or a video source file.
    """

    def __init__(self,
                 filename: str = None,
                 size: Tuple[int, int] = None,
                 camera_id: int = 0,
                 preserve_aspect_ratio: bool = True):
        """
        :param filename:
            Path to a video file.
        :param size:
            The expected frame size of the video.
        :param camera_id:
            Device index for the camera.
        :param preserve_aspect_ratio:
            Whether to preserve the aspect ratio of the video frames.
        """
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        if filename:
            self._cam = cv2.VideoCapture(filename)
        else:
            self._cam = cv2.VideoCapture(camera_id)
            self._cam.set(3, 480)  # Set frame width to 480
            self._cam.set(4, 640)  # Set frame height to 640

    def get_image(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Capture image from video stream frame-by-frame.
        The captured image and a scaled copy of the image are returned.
        """
        ret, img = self._cam.read()
        if ret:
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

    def get_fps(self) -> float:
        """Return the frame rate of the video source."""
        return self._cam.get(cv2.CAP_PROP_FPS)


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
