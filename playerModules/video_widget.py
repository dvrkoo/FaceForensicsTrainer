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
import cv2

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        self.video_writer = False
        super().__init__(parent)
        self.player = parent

        self.layout = QHBoxLayout(self)

        buttons_layout = QVBoxLayout()
        self.layout.addLayout(buttons_layout)

        self.model_dropdown = CheckableComboBox()
        self.model_dropdown.addItems(models_index)

        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.video_label)

        # Add widgets to the layouts
        buttons_layout.addWidget(self.model_dropdown)
        buttons_layout.setAlignment(Qt.AlignCenter)

        # Add play and pause buttons
        self.play_button = QPushButton("Pause", self)
        self.play_button.clicked.connect(parent.play_pause_video)
        self.play_button.hide()  # Initially hidden until a video is loaded
        buttons_layout.addWidget(self.play_button)

        # Add a button to open a file dialog
        self.load_button = QPushButton("Load Video/Image", self)
        self.load_button.clicked.connect(parent.load_media)
        buttons_layout.addWidget(self.load_button)

        # Add a button to unload models
        self.unload_button = QPushButton("Switch To Frequency Domain", self)
        self.unload_button.clicked.connect(parent.unload_models)
        buttons_layout.addWidget(self.unload_button)

        # Add the toggle button for switching between detection modes
        self.toggle_button = QPushButton("Switch to Image Detection", self)
        self.toggle_button.setCheckable(True)  # This makes it act as a toggle button
        self.toggle_button.setChecked(False)  # Initially set to image detection mode
        self.toggle_button.clicked.connect(self.switch_detection_mode)
        buttons_layout.addWidget(self.toggle_button)

        # Add the new button to extract frames
        self.extract_button = QPushButton("Extract Frames", self)
        self.extract_button.clicked.connect(self.extract_current_frame)
        buttons_layout.addWidget(self.extract_button)

    def process_video(self):
        self.player.video_writer = True

    def switch_detection_mode(self):
        # Toggle between image and video detection mode
        if self.toggle_button.isChecked():
            self.toggle_button.setText("Switch to Video Detection")
            self.load_button.setText("Load Image")
            self.player.detection_mode = "image"
            self.unload_button.hide()
            print(self.player.detection_mode)
        else:
            self.toggle_button.setText("Switch to Image Detection")
            self.unload_button.show()
            self.load_button.setText("Load Video")
            self.player.detection_mode = "video"

    def extract_current_frame(self):
        """Extract and save the current frame being displayed in the video."""
        if self.player.cap is None or not self.player.cap.isOpened():
            print("No video loaded or video capture is not open.")
            return

        # Get the current position in the video
        current_pos = int(self.player.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.player.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        # Read the current frame
        ret, frame = self.player.cap.read()
        if ret:
            # Save the current frame as an image file
            frame_filename = f"./ff_images/current_frame_{current_pos:04d}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved current frame as {frame_filename}")
        else:
            print("Failed to extract current frame.")
