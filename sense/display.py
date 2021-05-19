import cv2
import numpy as np
import time

from collections import deque
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

    def __init__(self, x_offset=350, y_offset=20):
        self.x_offset = x_offset
        self.y_offset = y_offset

    def initialize(self):
        """
        Called once the setup is done and the inference is ready to start.
        """
        pass

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

    def display(self, img, display_data):
        offset = 10
        for key in ['Met value', 'Total calories']:
            put_text(img, "{}: {:.1f}".format(key, display_data[key]), (offset, self.y_offset))
            offset += self.x_offset
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
                put_text(img, 'Proba: {:0.2f}'.format(proba), (10 + self.x_offset,
                                                               y_pos))
        return img


class DisplayCounts(BaseDisplay):

    def __init__(self, y_offset=40, highlight_changes=False, highlight_duration=3., fps=16, **kwargs):
        """
        :param y_offset:
            Vertical offset for showing the counts.
        :param highlight_changes:
            True if updated counts should be highlighted for some time.
        :param highlight_duration:
            The number of sections that an updated count should be highlighted.
        :param fps:
            The frame rate at which the display will be updated.
        """
        super().__init__(y_offset=y_offset, **kwargs)
        self.highlight_changes = highlight_changes
        self.previous_counts = deque(maxlen=int(highlight_duration * fps))

    def display_count(self, activity, count, img, y_pos, color):
        put_text(img, f'{activity}: {count}', (10 + self.x_offset, y_pos), color=color)

    def display(self, img, display_data):
        counters = display_data['counting']
        for index, (activity, count) in enumerate(counters.items()):
            if (self.highlight_changes
                    and self.previous_counts
                    and self.previous_counts[0].get(activity, 0) < count):
                color = (44, 176, 82)
            else:
                color = (255, 255, 255)
            y_pos = 20 * (index + 1) + self.y_offset
            self.display_count(activity, count, img, y_pos, color)

        self.previous_counts.append(counters)

        return img


class DisplayExerciseRepCounts(DisplayCounts):

    def display_count(self, activity, count, img, y_pos, color):
        put_text(img, f'Exercise: {activity[0:50]}', (10, y_pos), color=color)
        put_text(img, f'Count: {count}', (10 + self.x_offset, y_pos), color=color)


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
        self.last_update_time_camera = None
        self.last_update_time_inference = None

    def initialize(self):
        now = time.perf_counter()
        self.last_update_time_camera = now
        self.last_update_time_inference = now

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
            border_size_top: int = 50,
            border_size_right: int = 0,
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
        :param border_size_top:
            Height of the border on top of the video display. Used for correctly centering
            the displayed class name on the video.
        :param border_size_right:
            Width of the border added to the right of the video display. Used for correctly centering
            the displayed class name on the video.
        """
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.duration = duration
        self.font_scale = font_scale
        self.thickness = thickness
        self.border_size_top = border_size_top
        self.border_size_right = border_size_right

        self._current_class_name = None
        self._start_time = None

    def _get_center_coordinates(self, img: np.ndarray, text: str, font_scale: float) -> Tuple[int, int]:
        """
        Calculate and return the coordinates of the lower left corner of the text
        for centering it on the current frame.
        """
        text_size = cv2.getTextSize(text, FONT, font_scale, self.thickness)[0]

        height, width, _ = img.shape
        width -= self.border_size_right
        height -= self.border_size_top

        x = int((width - text_size[0]) / 2)
        y = int((height + text_size[1]) / 2) + self.border_size_top

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


class DisplayPredictionBarGraph(BaseDisplay):
    """
    Display the predicted class probabilities in a bar graph.
    """

    def __init__(
            self,
            keys: List[str],
            thresholds: Dict[str, float] = None,
            bar_length: int = 100,
            display_counts: bool = False,
            **kwargs,
    ):
        """
        :param keys:
            List of class names that should be displayed.
        :param thresholds:
            A dictionary specifying a threshold for each class name. The color of the corresponding
            bar in the graph will change when the probability passes the threshold.
        :param bar_length:
            Length of the bars in the bar graph.
        :param display_counts:
            Whether to display or not event counts (see e.g. postprocess.EventCounter) next
            to each bar.
        """

        super().__init__(**kwargs)
        self.keys = keys
        self.thresholds = thresholds or {}
        self.bar_length = bar_length
        self.display_counts = display_counts

    def display(self, img, display_data):
        results = dict(display_data['sorted_predictions'])

        font_scale = 1.2

        for index, key in enumerate(self.keys):
            proba = results[key]

            size_text = cv2.getTextSize(key, FONT, font_scale, 1)[0]
            x_text = self.x_offset - size_text[0]
            x_bar_left = self.x_offset + 10
            x_bar_right = x_bar_left + int(self.bar_length * proba)
            y_pos = 25 * (index + 1) + self.y_offset

            # determine color of the bar
            if proba > self.thresholds.get(key, 1.):
                bar_color = (0, 255, 0)  # bright green
            else:
                bar_color = (0, 128, 0)  # darker green

            # display key name
            put_text(img, key, (x_text, y_pos), font_scale=font_scale)

            # display bar next to key name
            cv2.line(img, (x_bar_left, y_pos - 5), (x_bar_right, y_pos - 5),
                     bar_color, 10, cv2.LINE_AA)

            # display proba value next to bar
            put_text(img, f"{proba:.2f}", (x_bar_right + 10, y_pos), font_scale=font_scale)

            # display event counts
            if self.display_counts:
                count = display_data['counting'][key]
                put_text(img, f"{count}", (x_bar_left + self.bar_length + 100, y_pos), font_scale=font_scale)

        return img


class DisplayResults:
    """
    Display window for an image frame with prediction outputs from a neural network.
    """
    def __init__(
            self,
            display_ops: List[BaseDisplay],
            title: Optional[str] = None,
            border_size_top: int = 30,
            border_size_right: int = 0,
            window_size: Tuple[int, int] = (480, 640),
            display_fn=None
    ):
        """
        :param display_ops:
            Additional options to be displayed on top of the image frame.
            Display options are class objects that implement the `display(self, img, display_data)` method.
            Current supported options include:
                - DisplayMETandCalories
                - DisplayDetailedMETandCalories
                - DisplayTopKClassificationOutputs
        :param title:
            Title of the image frame on display.
        :param border_size_top:
            Thickness of the top display border.
        :param border_size_right:
            Thickness of the right display border.
        :param window_size:
            Resolution of the display window.
        """
        self.window_size = window_size
        self._window_title = 'Real-time SenseNet'
        self.border_size_top = border_size_top
        self.border_size_right = border_size_right
        if not display_fn:
            cv2.namedWindow(self._window_title, cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self._window_title,
                             self.window_size[1] + self.border_size_right,
                             self.window_size[0] + self.border_size_top)
        self.title = title
        self.display_ops = display_ops
        self.display_fn = display_fn

    def initialize(self):
        """
        Initialize all contained display operations. Called once the setup is done and the inference is ready to start.
        """
        for display_op in self.display_ops:
            display_op.initialize()

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
        img = cv2.copyMakeBorder(img,
                                 self.border_size_top, 0,
                                 lr_pad, lr_pad + self.border_size_right,
                                 cv2.BORDER_CONSTANT)
        return img

    def clean_up(self):
        """Close all windows that are created."""
        cv2.destroyAllWindows()
