from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)
from .models_dropdown import CheckableComboBox

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setContentsMargins(0, 0, 0, 0)
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        buttons_layout = QVBoxLayout()
        self.layout.addLayout(buttons_layout)

        self.model_dropdown = CheckableComboBox()
        self.model_dropdown.addItems(models_index)

        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignLeft)

        # Add widgets to the layouts
        buttons_layout.addWidget(self.model_dropdown)
        self.layout.addWidget(self.video_label)
        buttons_layout.setAlignment(Qt.AlignCenter)

        # Add play and pause buttons
        self.play_button = QPushButton("Pause", self)
        self.play_button.clicked.connect(parent.play_pause_video)
        self.play_button.hide()  # Initially hidden until a video is loaded
        buttons_layout.addWidget(self.play_button)

        # Add a button to open a file dialog
        self.load_button = QPushButton("Load Video/Image", self)
        self.load_button.clicked.connect(parent.load_video)
        buttons_layout.addWidget(self.load_button)
