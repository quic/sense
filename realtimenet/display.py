import cv2
import numpy as np
import time

from typing import List
from typing import Tuple


FONT = cv2.FONT_HERSHEY_PLAIN


def put_text(
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 1,
        thickness: int = 1
) -> np.ndarray:
    """
    Draw a white text string on an image at a specified position and return the image.

    :param img:
        The image on which the text is to be drawn.
    :param text:
        The text to be written.
    :param position:
        A tuple of x and y coordinates of the bottom-left corner of the text in the image.
    :param font_scale:
        Font scale factor for modifying the font size.
    :param thickness:
        Thickness of the lines used to draw the text.

    :return:
        The image with the text string drawn.
    """
    cv2.putText(img, text, position, FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


class BaseDisplay:
    """
    Base display for all displays. Subclasses should overwrite the `display` method.
    """
    def __init__(self, y_offset=20):
        self.y_offset = y_offset

    def display(self, img: np.ndarray, display_data: dict) -> np.ndarray:
        """
        Method to be implemented by subclasses.
        This method writes display data onto an image frame.

        :param img:
            Image on which the display data should be written to.
        :param display_data:
            Data that should be displayed on an image frame.

        :return:
            The image with the display data written.
        """
        raise NotImplementedError


class DisplayMETandCalories(BaseDisplay):
    """
    Display Metabolic Equivalent of Task (MET) and Calories information on an image frame.
    """

    lateral_offset = 350

    def display(self, img, display_data):
        offset = 10
        for key in ['Met value', 'Total calories']:
            put_text(img, "{}: {:.1f}".format(key, display_data[key]), (offset, self.y_offset))
            offset += self.lateral_offset
        return img


class DisplayDetailedMETandCalories(BaseDisplay):
    """
    Display detailed Metabolic Equivalent of Task (MET) and Calories information on an image frame.
    """

    def display(self, img, display_data):
        offset = 10
        text = "MET (live): {:.1f}".format(display_data['Met value'])
        put_text(img, text, (offset, self.y_offset))
        offset += 175
        text = "MET (avg, corrected): {:.1f}".format(display_data['Corrected met value'])
        put_text(img, text, (offset, self.y_offset))
        offset += 275
        text = "CALORIES: {:.1f}".format(display_data['Total calories'])
        put_text(img, text, (offset, self.y_offset))
        return img


class DisplayTopKClassificationOutputs(BaseDisplay):
    """
    Display Top K Classification output on an image frame.
    """

    lateral_offset = DisplayMETandCalories.lateral_offset

    def __init__(self, top_k=1, threshold=0.2, **kwargs):
        """
        :param top_k:
            Number of the top classification labels to be displayed.
        :param threshold:
            Threshold for the output to be displayed.
        """
        super().__init__(**kwargs)
        self.top_k = top_k
        self.threshold = threshold

    def display(self, img, display_data):
        sorted_predictions = display_data['sorted_predictions']
        for index in range(self.top_k):
            activity, proba = sorted_predictions[index]
            y_pos = 20 * index + self.y_offset
            if proba >= self.threshold:
                put_text(img, 'Activity: {}'.format(activity[0:50]), (10, y_pos))
                put_text(img, 'Proba: {:0.2f}'.format(proba), (10 + self.lateral_offset,
                                                               y_pos))
        return img


class DisplayRepCounts(BaseDisplay):

    lateral_offset = DisplayMETandCalories.lateral_offset

    def __init__(self, y_offset=40):
        super().__init__(y_offset)

    def display(self, img, display_data):
        counters = display_data['counting']
        index = 0
        for activity, count in counters.items():
            y_pos = 20 * (index + 1) + self.y_offset
            put_text(img, 'Exercise: {}'.format(activity[0:50]), (10, y_pos))
            put_text(img, 'Count: {}'.format(count), (10 + self.lateral_offset, y_pos))
            index += 1
        return img


class DisplayDigits(BaseDisplay):
    """
    Display recognized digits as a large video overlay. Once the probability for a class passes the threshold,
    the digit is shown and stays visible for a certain duration.
    """

    def __init__(self, thresholds, duration=2, font_scale=20, thickness=10, border_size=100, **kwargs):
        """
        :param thresholds:
            Dictionary of thresholds for all classes.
        :param duration:
            Duration in seconds how long the digit should be displayed after it has been recognized.
        :param font_scale:
            Font scale factor for modifying the font size.
        :param thickness:
            Thickness of the lines used to draw the text.
        :param border_size:
            Height of the border on top of the video display. Used for correctly centering the displayed digit on the
            video.
        """
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.duration = duration
        self.font_scale = font_scale
        self.thickness = thickness
        self.border_size = border_size

        self._current_digit = None
        self._start_time = None

    def _get_center_coordinates(self, img, text):
        textsize = cv2.getTextSize(text, FONT, self.font_scale, self.thickness)[0]

        height, width, _ = img.shape
        height -= self.border_size

        x = int((width - textsize[0]) / 2)
        y = int((height + textsize[1]) / 2) + self.border_size

        return x, y

    def _display_digit(self, img, digit):
        pos = self._get_center_coordinates(img, digit)
        put_text(img, digit, pos, self.font_scale, self.thickness)

    def display(self, img, display_data):
        now = time.perf_counter()

        if self._current_digit and now - self._start_time < self.duration:
            # Keep displaying the same digit
            self._display_digit(img, self._current_digit)
        else:
            self._current_digit = None
            for class_name, proba in display_data['sorted_predictions']:
                if class_name in self.thresholds and proba > self.thresholds[class_name]:
                    # Display new digit
                    self._display_digit(img, class_name)
                    self._current_digit = class_name
                    self._start_time = now

                    break

        return img


class DisplayResults:
    """
    Display window for an image frame with prediction outputs from a neural network.
    """
    def __init__(self, title: str, display_ops: List[BaseDisplay], border_size: int = 50):
        """
        :param title:
            Title of the image frame on display.
        :param display_ops:
            Additional options to be displayed on top of the image frame.
            Display options are class objects that implement the `display(self, img, display_data)` method.
            Current supported options include:
                - DisplayMETandCalories
                - DisplayDetailedMETandCalories
                - DisplayTopKClassificationOutputs
        :param border_size:
            Thickness of the display border.
        """
        self._window_title = 'realtimenet'
        cv2.namedWindow(self._window_title, cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
        self.title = title
        self.display_ops = display_ops
        self.border_size = border_size

    def show(self, img: np.ndarray, display_data: dict) -> np.ndarray:
        """
        Show an image frame with data displayed on top.

        :param img:
            The image to be shown in the window.
        :param display_data:
            A dict of data that should be displayed in the image.

        :return:
            The image with displayed data.
        """
        # Mirror the img
        img = img[:, ::-1].copy()

        # Add black borders
        img = cv2.copyMakeBorder(img, self.border_size, 0, 0, 0, cv2.BORDER_CONSTANT)

        # Display information on top
        for display_op in self.display_ops:
            img = display_op.display(img, display_data)

        # Add title on top
        if self.title:
            img = cv2.copyMakeBorder(img, 50, 0, 0, 0, cv2.BORDER_CONSTANT)
            textsize = cv2.getTextSize(self.title, FONT, 1, 2)[0]
            middle = int((img.shape[1] - textsize[0]) / 2)
            put_text(img, self.title, (middle, 20))

        # Show the image in a window
        cv2.imshow(self._window_title, img)
        return img

    def clean_up(self):
        """Close all windows that are created."""
        cv2.destroyAllWindows()
