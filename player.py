import sys

import cv2
import torch
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import (
    QImage,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from playerModules import model_functions

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]
from playerModules.progressbar import ProgressBarWithTimeLabel
from playerModules.timer import VideoTimer

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


class VideoPlayerApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up UI
        self.setup_ui()
        self.models, self.detector = model_functions.load_models()

        # Initialize video capture to None
        self.cap = None
        self.timer = VideoTimer(self)
        self.timer.timeout_signal.connect(self.timerEvent)

        # Set default video dimensions
        self.video_width = 800
        self.video_height = 600

        # setup video playing state
        self.playing = False

        self.is_fake = False

    def setup_ui(self):
        # Set up layout
        layout = QVBoxLayout(self)

        # Add a button to open a file dialog
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        layout.addWidget(self.load_button)

        # Create a label for video display
        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        # Add play and pause buttons
        self.play_button = QPushButton("Pause", self)
        self.play_button.clicked.connect(self.play_pause_video)
        self.play_button.hide()
        layout.addWidget(self.play_button)

        # Progress bar
        self.progress_widget = ProgressBarWithTimeLabel(self)
        layout.addWidget(self.progress_widget)
        # Add predictions label
        self.predictions_label = QLabel(self)
        layout.addWidget(self.predictions_label)

        # Add bottom layout
        bottom_layout = QHBoxLayout()

        # Add additional widgets to the bottom layout if needed

        # Set the bottom layout for the main layout
        layout.addLayout(bottom_layout)

    def load_video(self):
        # Open a file dialog to choose a video file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi)"
        )

        if file_path:
            # If a file is selected, initialize video capture
            self.cap = cv2.VideoCapture(file_path)

            # set the default state to playing
            self.playing = True

            # show the play button
            self.play_button.show()

            # Check if the timer has been initialized before checking isActive
            if hasattr(self, "timer"):
                self.timer.start(33)  # ~30 fps
        self.load_button.hide()

        # Set the maximum value for the progress bar based on the video duration
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        video_duration = total_frames / fps * 1000
        self.progress_widget.progress_bar.setMaximum(int(video_duration))
        self.progress_widget.progress_bar.show()

    def play_pause_video(self):
        # Toggle between playing and pausing the video
        self.playing = not self.playing

        if self.playing:
            self.play_button.setText("Pause")
            self.timer.start(33)  # ~30 fps
        else:
            self.play_button.setText("Play")
            self.timer.stop()

    def timerEvent(self):
        # Read a frame from the video
        if self.playing:
            ret, frame = self.cap.read()

            if ret:
                frame = model_functions.convert_color_space(frame)
                # Perform face detection using dlib
                detected_faces = model_functions.detect_faces(frame, self.detector)
                if detected_faces:
                    # if more than one face is detected, use the first one which is also the biggest one
                    face_1 = detected_faces[0]
                    x, y, size = model_functions.get_boundingbox(face_1, frame)
                    face_roi = frame[y : y + size, x : x + size]

                    # Preprocess the face image for model input
                    input_tensor = model_functions.preprocess_input(face_roi)

                    # use model
                    (
                        prediction_value,
                        index,
                        fake,
                        predictions,
                    ) = model_functions.predict_with_model(input_tensor, self.models)
                    predictions_text = "Model Predictions:\n" + "\n".join(
                        [
                            f"{model}: {prediction:.4f}"
                            for model, prediction in zip(models_index, predictions)
                        ]
                    )
                    self.predictions_label.setText(predictions_text)
                    self.update_label(face_1, fake, index, frame)
                # Resize the frame to the fixed size
                frame_resized = cv2.resize(frame, (self.video_width, self.video_height))

                # Convert the OpenCV image to a QImage
                q_img = self.opencv_to_qimage(frame_resized)
                # Display the QImage in the QLabel
                pixmap = QPixmap.fromImage(q_img)
                self.video_label.setPixmap(pixmap)
                self.video_label.setScaledContents(True)

                # Update the progress bar based on the current time
                current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.progress_widget.progress_bar.setValue(int(current_time))
                self.progress_widget.update_time_label(current_time)
            else:
                # Stop the timer when the video ends
                self.playing = False
                self.play_button.setText("Play")
                self.load_button.show()
                self.kill_timer()

    def opencv_to_qimage(self, frame):
        # Convert the OpenCV image to a QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        return q_img

    def update_label(self, face, fake, index, frame):
        x, y, width, height = face.left(), face.top(), face.width(), face.height()

        self.is_fake = fake
        label = f"Faked with model: {models_index[index]}" if fake else "Genuine"
        label_color = (255, 0, 0) if fake else (0, 255, 0)

        cv2.putText(
            frame,
            label,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
        )

        cv2.rectangle(
            frame,
            (max(0, x), max(0, y)),
            (min(frame.shape[1], x + width), min(frame.shape[0], y + height)),
            (0, 255, 0),
            2,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    app = QApplication(sys.argv)
    window = VideoPlayerApp()
    window.show()
    sys.exit(app.exec_())