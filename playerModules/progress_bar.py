from PyQt5.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget


class ProgressBarWithTimeLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up layout
        layout = QHBoxLayout(self)

        # Add a progress bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Add a label for time display
        self.time_label = QLabel(self)
        layout.addWidget(self.time_label)

        # Set up style sheet
        self.setStyleSheet(
            "QProgressBar {text-align: center;}"
            "QProgressBar::chunk {background-color: white;}"
            "QLabel {margin-left: 5px;}"
        )

        # Connect signals and slots
        # self.progress_bar.valueChanged.connect(self.update_time_label)

    def set_frame_number(self, frame):
        self.total_frames = frame

    def update_time_label(self, value):
        curr_frame = (value / self.total_frames) * 100
        self.progress_bar.setValue(int(curr_frame))
        self.time_label.setText(f"{value}/{self.total_frames:02d}")
