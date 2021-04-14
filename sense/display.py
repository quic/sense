import cv2
import numpy as np
import time

from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional


FONT = cv2.FONT_HERSHEY_PLAIN


def put_text(
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 1.,
        color: Tuple[int, int, int] = (255, 255, 255),
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
    :param color:
        A tuple for font color. For BGR, eg: (0, 255, 0) for green color.
    :param thickness:
        Thickness of the lines used to draw the text.
    :return:
        The image with the text string drawn.
    """
    cv2.putText(img, text, position, FONT, font_scale, color, thickness, cv2.LINE_AA)
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


class DisplayFPS(BaseDisplay):
    """
    Display camera fps and inference engine fps on debug window.
    """

    def __init__(
            self,
            expected_camera_fps: Optional[float] = None,
            expected_inference_fps: Optional[float] = None,
            y_offset=10):
        super().__init__(y_offset)
        self.expected_camera_fps = expected_camera_fps
        self.expected_inference_fps = expected_inference_fps

        self.update_rate = 0.1
        self.low_performance_rate = 0.75
        self.default_text_color = (44, 176, 82)
        self.running_delta_time_inference = 1. / expected_inference_fps if expected_inference_fps else 0
        self.running_delta_time_camera = 1. / expected_camera_fps if expected_camera_fps else 0
        self.last_update_time_camera = time.perf_counter()
        self.last_update_time_inference = time.perf_counter()

    def display(self, img: np.ndarray, display_data: dict) -> np.ndarray:

        now = time.perf_counter()
        if display_data['prediction'] is not None:
            # Inference engine frame rate
            delta = (now - self.last_update_time_inference)
            self.running_delta_time_inference += self.update_rate * (delta - self.running_delta_time_inference)
            self.last_update_time_inference = now
        inference_engine_fps = 1. / self.running_delta_time_inference

        # Camera FPS counting
        delta = (now - self.last_update_time_camera)
        self.running_delta_time_camera += self.update_rate * (delta - self.running_delta_time_camera)
        camera_fps = 1. / self.running_delta_time_camera
        self.last_update_time_camera = now

        # Text color change if inference engine fps go below certain range
        if self.expected_inference_fps \
                and inference_engine_fps < self.expected_inference_fps * self.low_performance_rate:
            text_color = (0, 0, 255)
        else:
            text_color = self.default_text_color

        # Show FPS on the video screen
        put_text(img, "Camera FPS: {:.1f}".format(camera_fps), (5, img.shape[0] - self.y_offset - 20),
                 color=text_color)
        put_text(img, "Model FPS: {:.1f}".format(inference_engine_fps), (5, img.shape[0] - self.y_offset),
                 color=text_color)

        return img


class DisplayClassnameOverlay(BaseDisplay):
    """
    Display recognized class name as a large video overlay. Once the probability for a class passes the threshold,
    the name is shown and stays visible for a certain duration.
    """

    def __init__(
            self,
            thresholds: Dict[str, float],
            duration: float = 2.,
            font_scale: float = 3.,
            thickness: int = 2,
            border_size: int = 50,
            **kwargs
    ):
        """
        :param thresholds:
            Dictionary of thresholds for all classes.
        :param duration:
            Duration in seconds how long the class name should be displayed after it has been recognized.
        :param font_scale:
            Font scale factor for modifying the font size.
        :param thickness:
            Thickness of the lines used to draw the text.
        :param border_size:
            Height of the border on top of the video display. Used for correctly centering the displayed class name
            on the video.
        """
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.duration = duration
        self.font_scale = font_scale
        self.thickness = thickness
        self.border_size = border_size

        self._current_class_name = None
        self._start_time = None

    def _get_center_coordinates(self, img: np.ndarray, text: str, font_scale: float) -> Tuple[int, int]:
        """
        Calculate and return the coordinates of the lower left corner of the text
        for centering it on the current frame.
        """
        text_size = cv2.getTextSize(text, FONT, font_scale, self.thickness)[0]

        height, width, _ = img.shape
        height -= self.border_size

        x = int((width - text_size[0]) / 2)
        y = int((height + text_size[1]) / 2) + self.border_size

        return x, y

    def _adjust_font_scale(self, img: np.ndarray, text: str) -> float:
        """
        In case the class name text width with default font scale is larger than frame width,
        adjust font scale to fit the class name text within the frame width.
        """
        _, frame_width, _ = img.shape
        text_width = cv2.getTextSize(text, FONT, self.font_scale, self.thickness)[0][0]
        text_to_frame_ratio = text_width / frame_width

        if text_to_frame_ratio > 1:
            return self.font_scale / text_to_frame_ratio
        return self.font_scale

    def _display_class_name(self, img: np.ndarray, class_name: str) -> None:
        """
        Display the class name on the center of the current frame.
        """
        font_scale = self._adjust_font_scale(img=img, text=class_name)
        pos = self._get_center_coordinates(img=img, text=class_name, font_scale=font_scale)
        put_text(img, class_name, position=pos, font_scale=font_scale, thickness=self.thickness)

    def display(self, img: np.ndarray, display_data: dict) -> np.ndarray:
        """
        Render all display data onto the current frame.
        """
        now = self._get_current_time()

        if self._current_class_name and now - self._start_time < self.duration:
            # Keep displaying the same class name
            self._display_class_name(img, self._current_class_name)
        else:
            self._current_class_name = None
            for class_name, proba in display_data['sorted_predictions']:
                if class_name in self.thresholds and proba > self.thresholds[class_name]:
                    # Display new class name
                    self._display_class_name(img, class_name)
                    self._current_class_name = class_name
                    self._start_time = now

                    break

        return img

    @staticmethod
    def _get_current_time() -> float:
        """
        Wrapper method to get the current time.
        Extracted for ease of testing.
        """
        return time.perf_counter()


class DisplayResults:
    """
    Display window for an image frame with prediction outputs from a neural network.
    """
    def __init__(self, title: str, display_ops: List[BaseDisplay], border_size: int = 30,
                 window_size: Tuple[int, int] = (480, 640), display_fn=None):
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
        :param window_size:
            Resolution of the display window.
        """
        self.window_size = window_size
        self._window_title = 'Real-time SenseNet'
        self.border_size = border_size
        if not display_fn:
            cv2.namedWindow(self._window_title, cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self._window_title, self.window_size[1], self.window_size[0] + self.border_size)
        self.title = title
        self.display_ops = display_ops
        self.display_fn = display_fn

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

        # Adjust image to fit in display window
        img = self.resize_to_fit_window(img)

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
        if self.display_fn:
            self.display_fn(img)
        else:
            cv2.imshow(self._window_title, img)
        return img

    def resize_to_fit_window(self, img):
        height, width = img.shape[0:2]
        window_aspect_ratio = self.window_size[1] / self.window_size[0]
        img_aspect_ratio = width / height

        if img_aspect_ratio < window_aspect_ratio:
            new_height = self.window_size[0]
            new_width = round(new_height * width / height)
        else:
            new_width = self.window_size[1]
            new_height = round(new_width * height / width)

        img = cv2.resize(img, (new_width, new_height))
        # Pad black borders:
        #   - top: controlled by border_size
        #   - bottom: none
        #   - left-right: so that the width is equal to window_size[1]
        lr_pad = max(round((self.window_size[1] - new_width) / 2), 0)
        img = cv2.copyMakeBorder(img, self.border_size, 0, lr_pad, lr_pad, cv2.BORDER_CONSTANT)
        return img

    def clean_up(self):
        """Close all windows that are created."""
        cv2.destroyAllWindows()
