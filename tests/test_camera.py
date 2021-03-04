import unittest
from unittest.mock import patch

import cv2
import numpy as np
import os

import sense.camera as camera

VIDEO_PATH = os.path.join(os.getcwd(), 'tests', 'resources', 'test_video.mp4')


class TestVideoSource(unittest.TestCase):

    def setUp(self) -> None:
        self.video = camera.VideoSource(filename=VIDEO_PATH)

    @patch('sense.camera.VideoSource.pad_to_square')
    def test_get_image(self, mock_pad_to_square):
        square_img_path = os.path.join(os.getcwd(), 'tests', 'resources', 'square_SENSE.png')
        square_img = cv2.imread(square_img_path)
        mock_pad_to_square.return_value = square_img
        img, scaled_img = self.video.get_image()
        self.assertTrue(mock_pad_to_square)
        assert img.shape == scaled_img.shape

    def test_pad_to_square(self):
        img_path = os.path.join(os.getcwd(), 'tests', 'resources', 'SENSE.png')
        img = cv2.imread(img_path)
        max_length = max(img.shape[0:2])
        new_img = self.video.pad_to_square(img)
        assert new_img.shape == (max_length, max_length, 3)

    def test_fps(self):
        fps = self.video.get_fps()
        assert fps == 12.0


class TestVideoStream(unittest.TestCase):

    def setUp(self) -> None:
        self.video = camera.VideoSource(filename=VIDEO_PATH)
        self.stream = camera.VideoStream(video_source=self.video, fps=12.0)

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
        self.output_video_path = os.path.join(os.getcwd(), 'tests', 'resources', 'test_writer.mp4')
        self.videowriter = camera.VideoWriter(path=self.output_video_path, fps=12.0, resolution=(40, 30))

    def test_write(self):
        input_video_path = VIDEO_PATH
        input_video = cv2.VideoCapture(input_video_path)
        output_video_path = self.output_video_path

        while True:
            ret, frame = input_video.read()
            self.videowriter.write(frame)
            if not ret:
                break

        input_video.release()
        self.videowriter.release()
        output_video = cv2.VideoCapture(output_video_path)
        self.assertTrue(output_video.isOpened())

    def test_release(self):
        self.videowriter.release()
        self.assertFalse(self.videowriter.writer.isOpened())


if __name__ == '__main__':
    unittest.main()
