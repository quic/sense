import os
import unittest
from unittest.mock import patch
import numpy as np

from sense.camera import uniform_frame_sample
from sense.camera import VideoSource


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

    VIDEO_FILE = os.path.join(os.path.dirname(__file__), 'resources', 'test_video.mp4')

    def test_from_file(self):
        video_source = VideoSource(filename=self.VIDEO_FILE)
        assert video_source._frames is None
        assert video_source.get_image() is not None

    def test_from_file_change_fps(self):
        video_source = VideoSource(filename=self.VIDEO_FILE, target_fps=5)
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
