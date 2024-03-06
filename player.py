import sys

import cv2
import torch
from PyQt5.QtGui import (
    QImage,
    QPixmap,
)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSizePolicy,
    QScrollArea,
    QCheckBox,
)


from playerModules import model_functions, prediction_bar
from playerModules.models_dropdown import CheckableComboBox
from playerModules.progress_bar import ProgressBarWithTimeLabel
from playerModules.timer import VideoTimer

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


class VideoPlayerApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up UI
        self.setup_ui()
        self.models, self.detector = model_functions.load_models()
        self.selected_models = None
        self.resizing = False
        # Initialize video capture to None
        self.cap = None
        self.timer = VideoTimer(self)
        self.timer.timeout_signal.connect(self.timerEvent)

        # Set default video dimensions
        self.video_width = 600
        self.video_height = 800

        # setup video playing state
        self.playing = False

        self.is_fake = False

        # setup prediction bar

    def model_selection_changed(self, index):
        # Slot to be triggered when the selected model changes
        selected_model = models_index[index - 1] if index > 0 else None
        self.selected_model = selected_model  # Store the selected model in the class

    def setup_ui(self):
        # Set up layout
        layout = QVBoxLayout(self)
        self.model_dropdown = CheckableComboBox()
        self.model_dropdown.addItems(models_index)
        layout.addWidget(self.model_dropdown)
        selected_models = self.model_dropdown.returnSelectedItems()
        self.video_label = QLabel(self)
        # self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)
        # Add play and pause buttons
        self.play_button = QPushButton("Pause", self)
        self.play_button.clicked.connect(self.play_pause_video)
        self.play_button.hide()
        layout.addWidget(self.play_button)

        # Add a button to open a file dialog
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        layout.addWidget(self.load_button)

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

        # add prediction bar
        self.prediction_bar = prediction_bar.PredictionsBarGraph(self)
        # Create a QScrollArea
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)  # Allow prediction_bar to expand
        scroll_area.setWidget(self.prediction_bar)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Add
        # Add the scroll area to your layout
        layout.addWidget(scroll_area)
        # layout.addWidget(self.prediction_bar)

    def load_video(self):
        # Open a file dialog to choose a video file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi)"
        )

        if file_path:
            # If a file is selected, initialize video capture
            self.cap = cv2.VideoCapture(file_path)
            self.prediction_bar.past_predictions = [[] for _ in range(5)]

            # Get video dimensions
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # self.video_label.setFixedSize(self.video_width, self.video_height)

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
            self.load_button.hide()
        else:
            self.play_button.setText("Play")
            self.load_button.show()
            self.timer.stop()

    def update_predictions(self, predictions, selected_model):
        self.prediction_bar.set_predictions(
            predictions, models_index, self.selected_models
        )

    # TODO delete this function
    def setup_predictions_text(self, predictions):
        if self.selected_model is None:
            self.predictions_label.setText(
                "Model Predictions:\n"
                + "\n".join(
                    [
                        f"{model}: {prediction:.4f}"
                        for model, prediction in zip(models_index, predictions)
                    ]
                )
            )
        else:
            self.predictions_label.setText(
                f"Model Predictions:\n{self.selected_model}: {predictions[0]:.4f}"
            )

    def timerEvent(self):
        self.selected_models = self.model_dropdown.returnSelectedItems()
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
                    indexes, fakes, predictions = ([] for _ in range(3))
                    for selected_model in self.selected_models:
                        # use model
                        (
                            prediction_value,
                            index,
                            fake,
                            prediction,
                        ) = model_functions.predict_with_model(
                            input_tensor,
                            self.models,
                            selected_model=selected_model,
                        )
                        indexes.append(index)
                        fakes.append(fake)
                        predictions.append(prediction)

                    self.update_predictions(predictions, self.selected_models)
                    # self.setup_predictions_text(predictions)
                    self.update_label(face_1, max(fakes), index, frame)
                # Calculate scaling factors
                height_scale = self.video_label.height() / frame.shape[0]
                width_scale = self.video_label.width() / frame.shape[1]
                scale = min(height_scale, width_scale)

                # Resize the frame to fit the window
                frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

                # Convert the OpenCV image to a QImage
                q_img = self.opencv_to_qimage(frame_resized)
                # Display the QImage in the QLabel
                pixmap = QPixmap.fromImage(q_img)
                self.video_label.setPixmap(pixmap)
                self.video_label.setAlignment(Qt.AlignCenter)

                # Resize the window to fit the scaled-down video
                # self.resize(pixmap.width(), pixmap.height())

                # Update the progress bar based on the current time
                current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.progress_widget.progress_bar.setValue(int(current_time))
                self.progress_widget.update_time_label(current_time)
            else:
                # Stop the timer when the video ends
                self.playing = False
                self.play_button.setText("Play")
                self.load_button.show()
                self.timer.stop()

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
            label_color,
            2,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)
    window = VideoPlayerApp()
    window.show()
    sys.exit(app.exec_())
