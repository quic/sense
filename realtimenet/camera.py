import queue
import time

import cv2


from threading import Thread


class VideoSource:
    """
    Grabs frames from a camera or a video file.
    """

    def __init__(self, filename=None, size=None, camera_id=0, preserve_aspect_ratio=True):
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        if filename:
            self._cam = cv2.VideoCapture(filename)
        else:
            self._cam = cv2.VideoCapture(camera_id)
            self._cam.set(3, 480)
            self._cam.set(4, 640)

    def get_image(self):
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
        square_size = max(img.shape[0:2])
        pad_top = int((square_size - img.shape[0]) / 2)
        pad_bottom = square_size - img.shape[0] - pad_top
        pad_left = int((square_size - img.shape[1]) / 2)
        pad_right = square_size - img.shape[1] - pad_left
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    def get_sample_rate(self):
        return self._cam.get(cv2.CAP_PROP_FPS)


class VideoStream(Thread):
    """
    Thread that reads frames from the video source at a given framerate.
    """

    def __init__(self, video_source, fps, queue_size=4):
        Thread.__init__(self)
        self.video_source = video_source
        self.frames = queue.Queue(queue_size)
        self.fps = fps
        self.delta_t = 1.0 / self.fps
        self._shutdown = False

    def stop(self):
        self._shutdown = True

    def get_image(self):
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

    def __init__(self, path, fps, resolution):
        self.writer = cv2.VideoWriter(path, 0x7634706d, fps, resolution)
        self.delta_t = 1.0 / fps
        self._last_time = -1

    def write(self, frame):
        now = time.perf_counter()
        if now - self._last_time >= self.delta_t:
            self.last_time_written = now
            self.writer.write(frame)

    def release(self):  # noqa: D102
        self.writer.release()


