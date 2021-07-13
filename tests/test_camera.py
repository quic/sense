import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sense.camera import uniform_frame_sample
from sense.camera import VideoSource
from sense.camera import VideoStream
from sense.camera import VideoWriter

VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'test_video.mp4')


class TestUniformFrameSample(unittest.TestCase):

    VIDEO = np.array([1, 2, 3, 4, 5, 6])

    def test_upsampling(self):
        upsampled_video = uniform_frame_sample(self.VIDEO, 1.5)
        assert np.array_equal(upsampled_video, [1, 2, 2, 3, 3, 4, 5, 5, 6])

    def test_downsampling(self):
        downsampled_video = uniform_frame_sample(self.VIDEO, 0.5)
        assert np.array_equal(downsampled_video, [2, 3, 5])

    def test_no_change(self):
        same_video = uniform_frame_sample(self.VIDEO, 1.0)
        assert np.array_equal(same_video, self.VIDEO)


class TestVideoSource(unittest.TestCase):

    def test_from_file(self):
        video_source = VideoSource(filename=VIDEO_PATH)
        assert video_source._frames is None
        assert video_source.get_image() is not None

    def test_from_file_change_fps(self):
        video_source = VideoSource(filename=VIDEO_PATH, target_fps=5)
        assert video_source._frames is not None  # Frames should be pre-computed and resampled
        assert video_source.get_image() is not None

    def test_from_camera(self):
        class MockVideoSource:
            def __init__(self, *args):
                pass

            def set(self, *args):
                pass

            def read(self):
                return True, np.zeros((2, 3))

        with patch('cv2.VideoCapture', MockVideoSource):
            video_source = VideoSource(camera_id=0)
        assert video_source._frames is None
        assert video_source.get_image() is not None


class TestVideoStream(unittest.TestCase):

    def setUp(self) -> None:
        self.video = VideoSource(filename=VIDEO_PATH)
        self.stream = VideoStream(video_source=self.video, fps=12.0)

    def test_stop(self):
        self.stream.stop()
        self.assertTrue(self.stream._shutdown)

    def test_extract_image(self):
        img_tuple = self.video.get_image()
        self.stream.frames.put(img_tuple, False)
        imgs = self.stream.get_image()
        assert type(imgs[0]) == np.ndarray

    def test_frame_conditions(self):
        self.stream.run()
        self.assertTrue(self.stream.frames.full())
        self.assertTrue(self.stream._shutdown)


class TestVideoWriter(unittest.TestCase):

    def setUp(self) -> None:
        self.output_video_path = os.path.join(os.path.dirname(__file__), 'resources', 'test_writer.mp4')
        self.videowriter = VideoWriter(path=self.output_video_path, fps=12.0, resolution=(40, 30))

    def test_write(self):
        input_video = cv2.VideoCapture(VIDEO_PATH)

        while True:
            ret, frame = input_video.read()
            self.videowriter.write(frame)
            if not ret:
                break

        input_video.release()
        self.videowriter.release()
        output_video = cv2.VideoCapture(self.output_video_path)
        # Open input video again to compare frames
        input_video = cv2.VideoCapture(VIDEO_PATH)

        # Check whether new output file is the same as the original
        while True:
            ret, frame_in = input_video.read()
            _, frame_out = output_video.read()
            if not ret:
                break
            assert np.allclose(frame_in, frame_out, atol=50)

        # Delete output file on test completion
        os.remove(self.output_video_path)

    def test_release(self):
        self.videowriter.release()
        self.assertFalse(self.videowriter.writer.isOpened())


if __name__ == '__main__':
    unittest.main()
