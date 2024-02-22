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
            "QProgressBar::chunk {background-color: green;}"
            "QLabel {margin-left: 5px;}"
        )

        # Connect signals and slots
        self.progress_bar.valueChanged.connect(self.update_time_label)

    def update_time_label(self, value):
        # Update the time label based on the progress bar value
        total_seconds = value / 1000
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        self.time_label.setText(f"{minutes}:{seconds:02d}")
