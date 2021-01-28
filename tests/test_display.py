import unittest
from unittest.mock import patch

import numpy as np

import sense.display as base_display


class TestDisplayMETandCalories(unittest.TestCase):

    def setUp(self) -> None:
        self.img = None
        self.display_data = {'Met value': 2.0, 'Total calories': 10.0}

    @patch('sense.display.put_text')
    def test_display(self, mock_put_text):
        test_display = base_display.DisplayMETandCalories()
        test_display.display(self.img, self.display_data)
        mock_put_text.assert_called_with(self.img, 'Total calories: 10.0', (360, 20))


class TestDisplayDetailedMETandCalories(unittest.TestCase):

    def setUp(self) -> None:
        self.img = None
        self.display_data = {'Met value': 2.0, 'Corrected met value': 3.0,
                             'Total calories': 10.0}

    @patch('sense.display.put_text')
    def test_display(self, mock_put_text):
        test_display = base_display.DisplayDetailedMETandCalories()
        test_display.display(self.img, self.display_data)
        mock_put_text.assert_called_with(self.img, 'CALORIES: 10.0', (460, 20))


class TestDisplayTopKClassificationOutputs(unittest.TestCase):

    def setUp(self) -> None:
        self.img = None
        self.display_data = {'sorted_predictions': [['Jumping Jacks', 0.5]]}

    @patch('sense.display.put_text')
    def test_display(self, mock_put_text):
        test_display = base_display.DisplayTopKClassificationOutputs()
        test_display.display(self.img, self.display_data)
        mock_put_text.assert_called_with(self.img, 'Proba: 0.50', (360, 20))


class TestDisplayRepCounts(unittest.TestCase):

    def setUp(self) -> None:
        self.img = None
        self.display_data = {'counting': {'Jumping Jacks': 10}}

    @patch('sense.display.put_text')
    def test_display(self, mock_put_text):
        test_display = base_display.DisplayRepCounts()
        test_display.display(self.img, self.display_data)
        mock_put_text.assert_called_with(self.img, 'Count: 10', (360, 60))


class TestDisplayResults(unittest.TestCase):

    def test_show_on_small_image(self):
        height, width = 300, 400
        img = np.ones(shape=(height, width, 3))
        test_show = base_display.DisplayResults(title="Demo", display_ops=[])
        resized_img = test_show.resize_to_fit_window(img)
        assert test_show.window_size[1] == resized_img.shape[1]

    def test_show_on_large_image(self):
        height, width = 720, 1280
        img = np.ones(shape=(height, width, 3))
        test_show = base_display.DisplayResults(title="Demo", display_ops=[])
        resized_img = test_show.resize_to_fit_window(img)
        assert test_show.window_size[1] == resized_img.shape[1]

    def test_show_on_equal_image(self):
        height, width = 480, 640
        img = np.ones(shape=(height, width, 3))
        test_show = base_display.DisplayResults(title="Demo", display_ops=[])
        resized_img = test_show.resize_to_fit_window(img)
        assert img.shape[0] + test_show.border_size == resized_img.shape[0]


if __name__ == '__main__':
    unittest.main()
