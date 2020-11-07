import cv2
from typing import Any
from typing import List
from typing import Tuple


FONT = cv2.FONT_HERSHEY_PLAIN


def put_text(img: Any, text: str, position: Tuple[int, int]) -> Any:
    """
    Draw a white text string on an image at a specified position and return the image.

    :param img:
        The image on which the text is to be drawn.
    :param text:
        The text to be written.
    :param position:
        A tuple of x and y coordinates of the bottom-left corner of the text in the image.

    :return:
        The image with the text string drawn.
    """
    cv2.putText(img, text, position, FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class DisplayResults:
    """
    Display window for an image frame with prediction outputs from a neural network.
    """
    def __init__(self, title: str, display_ops: List[Any], border_size: int = 50):
        """
        :param title:
            Title of the image frame on display.
        :param display_ops:
            Additional options to be displayed on top of the image frame.
            Display options are class objects that implement the `display(self, img, display_data)` method.
            Current supported options include:
                - DisplayMETandCalories
                - DisplayDetailedMETandCalories
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

    def show(self, img: Any, display_data: dict) -> Any:
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


class DisplayMETandCalories:

    lateral_offset = 350

    def __init__(self, y_offset=20):
        self.y_offset = y_offset

    def display(self, img, display_data):
        offset = 10
        for key in ['Met value', 'Total calories']:
            put_text(img, "{}: {:.1f}".format(key, display_data[key]), (offset, self.y_offset))
            offset += self.lateral_offset
        return img


class DisplayDetailedMETandCalories:

    def __init__(self, y_offset=20):
        self.y_offset = y_offset

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


class DisplayTopKClassificationOutputs:

    lateral_offset = DisplayMETandCalories.lateral_offset

    def __init__(self, top_k=1, threshold=0.2, y_offset=20):
        self.top_k = top_k
        self.threshold = threshold
        self.y_offset = y_offset

    def display(self, img, display_data):
        sorted_predictions = display_data['sorted_predictions']
        for index in range(self.top_k):
            activity, proba = sorted_predictions[index]
            y_pos = 20 * (index + 1) + self.y_offset
            if proba >= self.threshold:
                put_text(img, 'Activity: {}'.format(activity[0:50]), (10, y_pos))
                put_text(img, 'Proba: {:0.2f}'.format(proba), (10 + self.lateral_offset,
                                                               y_pos))
        return img
