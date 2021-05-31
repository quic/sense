import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from cv2 import FONT_HERSHEY_PLAIN
from cv2 import getTextSize
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
        test_display = base_display.DisplayExerciseRepCounts()
        test_display.display(self.img, self.display_data)
        mock_put_text.assert_called_with(self.img, 'Count: 10', (360, 60), color=(255, 255, 255))


class TestDisplayClassnameOverlay(unittest.TestCase):

    def setUp(self) -> None:
        self.img = Mock()
        self.img.shape = (510, 640, 3)
        self.font_scale = 3.0
        self.thickness = 2
        self.duration = 2
        self.half_duration = self.duration / 2

    @patch('sense.display.put_text')
    def test_display_below_threshold(self, mock_put_text):
        test_display = base_display.DisplayClassnameOverlay(thresholds={'Dabbing': 0.5})
        test_display.display(self.img, {'sorted_predictions': [['Dabbing', 0.4]]})

        mock_put_text.assert_not_called()

    @patch('sense.display.put_text')
    def test_display_class_not_in_threshold(self, mock_put_text):
        test_display = base_display.DisplayClassnameOverlay(thresholds={'Dabbing': 0.5})
        test_display.display(self.img, {'sorted_predictions': [['Nodding', 1.0]]})

        mock_put_text.assert_not_called()

    @patch('sense.display.put_text')
    def test_display_default(self, mock_put_text):
        test_display = base_display.DisplayClassnameOverlay(thresholds={'Dabbing': 0.5})
        test_display.display(self.img, {'sorted_predictions': [['Dabbing', 0.6]]})

        mock_put_text.assert_called_with(self.img,
                                         'Dabbing',
                                         font_scale=self.font_scale,
                                         position=(224, 294),
                                         thickness=self.thickness)

    @patch('sense.display.DisplayClassnameOverlay._display_class_name')
    def test_display_default_within_duration(self, mock_display_class_name):
        test_display = base_display.DisplayClassnameOverlay(thresholds={'Dabbing': 0.5},
                                                            duration=self.duration)

        # Set start time and call display once
        test_display.display(self.img, {'sorted_predictions': [['Dabbing', 0.51]]})
        mock_display_class_name.assert_called_with(self.img, 'Dabbing')
        assert mock_display_class_name.call_count == 1

        # Class name will continue to be displayed during the duration even if input is below threshold
        with patch("sense.display.DisplayClassnameOverlay._get_current_time",
                   return_value=test_display._start_time + self.half_duration):
            test_display.display(self.img, {'sorted_predictions': [['Dabbing', 0.49]]})
            assert mock_display_class_name.call_count == 2

        # Call display post the duration
        with patch("sense.display.DisplayClassnameOverlay._get_current_time",
                   return_value=test_display._start_time + self.duration):
            # Display nothing after duration has passed
            test_display.display(self.img, {'sorted_predictions': [['Dabbing', 0.49]]})
            # Total call counts should remain the same
            assert mock_display_class_name.call_count == 2

    @patch('sense.display.put_text')
    def test_display_adjust_font_scale(self, mock_put_text):
        test_display = base_display.DisplayClassnameOverlay(thresholds={'Swiping down (with two hands)': 0.5})
        test_display.display(self.img, {'sorted_predictions': [['Swiping down (with two hands)', 0.6]]})

        text_width = getTextSize('Swiping down (with two hands)', FONT_HERSHEY_PLAIN, self.font_scale,
                                 self.thickness)[0][0]
        _, frame_width, _ = self.img.shape
        font_scale = self.font_scale / (text_width / frame_width)

        mock_put_text.assert_called_with(self.img,
                                         'Swiping down (with two hands)',
                                         font_scale=font_scale,
                                         position=(0, 291),
                                         thickness=self.thickness)


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
        assert img.shape[0] + test_show.border_size_top == resized_img.shape[0]


if __name__ == '__main__':
    unittest.main()
