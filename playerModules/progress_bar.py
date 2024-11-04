from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget
from PyQt5.QtCore import Qt
import cv2


class ProgressBarWithTimeLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grandparent_widget = parent.parent()  # Access the grandparent widget

        # Set up layout
        layout = QHBoxLayout(self)

        # Add a slider
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)  # Adjust as necessary
        layout.addWidget(self.slider)

        # Add a label for time display
        self.time_label = QLabel(self)
        layout.addWidget(self.time_label)

        # Set up style sheet
        self.setStyleSheet("QLabel {margin-left: 5px;}")

        # Connect the slider to update the time label
        self.slider.valueChanged.connect(self.update_time_label)

        # Connect the sliderMoved signal to seek to the frame
        self.slider.sliderMoved.connect(self.seek_to_frame)

    def set_frame_number(self, total_frames):
        self.total_frames = total_frames
        self.slider.setMaximum(total_frames - 1)  # Slider range: 0 to total_frames - 1

    def update_time_label(self, value):
        # Update the time label to show the current frame and total frames
        self.time_label.setText(f"{value + 1}/{self.total_frames}")

        # Update the slider position to match the current frame
        self.slider.blockSignals(
            True
        )  # Temporarily block signals to prevent feedback loops
        self.slider.setValue(value)  # Update slider position
        self.slider.blockSignals(False)  # Re-enable signals

    def seek_to_frame(self, frame_number):
        if self.grandparent_widget and hasattr(self.grandparent_widget, "cap"):
            # Seek to the specified frame using OpenCV
            self.grandparent_widget.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read and display the frame
            self.grandparent_widget.current_frame = frame_number
            ret, frame = self.grandparent_widget.cap.read()
            if ret:
                # Convert the frame to RGB and display it
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.grandparent_widget.display_frame(frame_rgb)
