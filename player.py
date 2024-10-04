import sys
import cv2
import torch
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
)


from playerModules import model_functions
from playerModules.timer import VideoTimer
from Trufor.src.trufor_test import load_model
from Trufor.visualize import ProcessedImageWidget
from playerModules.video_widget import VideoWidget
from playerModules.logo import HomeScreenWidget
from playerModules.video_predictions import VideoPredictionWidget

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


class VideoPlayerApp(QWidget):
    def __init__(self):
        super().__init__()
        # Set up UI
        self.setup_ui()
        self.models, self.detector = model_functions.load_models()
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
        self.home_widget = HomeScreenWidget(self)
        self.layout.addWidget(self.home_widget)
        self.video_prediction_widget = None

    def setup_ui(self):
        # Set up layout
        self.processed_image_widget = None
        self.layout = QVBoxLayout(self)
        self.video_widget = VideoWidget(self)
        self.video_widget.hide()
        self.layout.addWidget(self.video_widget, stretch=1)

    def setup_video_prediction(self):
        """Initializes the video prediction widget and adds it to the main layout."""
        self.video_prediction_widget = VideoPredictionWidget(self)
        self.layout.addWidget(self.video_prediction_widget)

    def update_predictions(self, predictions):
        """Updates the predictions in the video prediction widget."""
        if hasattr(self, "video_prediction_widget"):
            self.video_prediction_widget.update_predictions_texts(predictions)
            self.video_prediction_widget.update_predictions(
                predictions, self.selected_models
            )

    def model_selection_changed(self, index):
        # Slot to be triggered when the selected model changes
        selected_model = models_index[index - 1] if index > 0 else None
        self.selected_model = selected_model  # Store the selected model in the class

    def setup_image_prediction(self):
        pass

    def clear_logo_widget(self):
        """Clear the home screen widget."""
        if hasattr(self, "home_widget"):
            self.home_widget.clear()  # Clear the home screen widgets
            self.layout.removeWidget(self.home_widget)
            self.home_widget.deleteLater()
            del self.home_widget  # Optional: delete reference to avoid future errors

    def load_video(self):
        # Open a file dialog to choose a video or image file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Open Video/Image File",
            "",
            "Video Files (*.mp4 *.avi *.jpg *.png)",
        )
        if file_path:

            self.clear_logo_widget()
            self.video_widget.show()
            # If a file is selected, initialize video capture
            self.cap = cv2.VideoCapture(file_path)
            # Get the total number of frames in the video (1 for an image)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Check if it's an image by looking at file extension
            if file_path.endswith((".jpg", ".png")):

                if self.processed_image_widget:
                    self.layout.removeWidget(self.processed_image_widget)
                    self.processed_image_widget.deleteLater()  # Safely delete the old widget

                image = cv2.imread(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.cap = image_rgb
                self.image = True
                # self.setup_image_prediction()
                if self.video_prediction_widget:
                    self.video_prediction_widget.hide()
                load_model(file_path)
                self.processed_image_widget = ProcessedImageWidget(file_path)
                self.layout.addWidget(self.processed_image_widget)
            else:
                self.image = False
                if self.processed_image_widget:
                    self.processed_image_widget.hide()
                if not self.video_prediction_widget:
                    self.setup_video_prediction()
                else:
                    self.video_prediction_widget.reset_past_predictions()
                self.video_prediction_widget.show()
                self.current_frame = 0
                ret, frame = self.cap.read()
                self.video_prediction_widget.progress_bar.set_frame_number(
                    self.total_frames
                )
                if ret:
                    self.display_frame(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )  # Show the first frame
                else:
                    self.playing = False
                # self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Show the play button and reset the state
            self.playing = False
            self.video_widget.play_button.show()
            self.video_widget.play_button.setText("Play")
            # Start the video timer
            if hasattr(self, "timer"):
                self.timer.start(33)  # ~30 fps for video

    def play_pause_video(self):
        # Toggle between playing and pausing the video
        self.playing = not self.playing

        if self.playing:
            self.video_widget.play_button.setText("Pause")
            self.timer.start(33)  # ~30 fps
        else:
            self.video_widget.play_button.setText("Play")
            self.timer.stop()

    def timerEvent(self):
        self.selected_models = self.video_widget.model_dropdown.returnSelectedItems()
        # Read a frame from the video
        if self.playing and not self.image:
            self.current_frame += 1
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
                    predictions = []
                    for model in models_index:
                        # use model
                        if model not in self.selected_models:
                            predictions.append([0])
                            continue
                        else:
                            prediction = model_functions.predict_with_model(
                                input_tensor,
                                self.models,
                                selected_model=model,
                            )
                            predictions.append(prediction)

                    self.update_label(face_1, predictions, frame)
                    self.update_predictions(predictions)
                self.display_frame(frame)
            else:
                self.playing = False
                self.video_widget.play_button.setText("Play")
                self.load_button.show()
                self.timer.stop()
        # frame logic
        elif self.image:
            frame = self.cap  # Calculate scaling factors
            self.display_frame(frame)

    def display_frame(self, frame):
        # Calculate scaling factors
        height_scale = self.video_widget.video_label.height() / frame.shape[0]
        width_scale = self.video_widget.video_label.width() / frame.shape[1]
        scale = min(height_scale, width_scale)

        # Resize the frame to fit the window
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

        # Convert the OpenCV image to a QImage
        q_img = self.opencv_to_qimage(frame_resized)

        # Display the QImage in the QLabel
        pixmap = QPixmap.fromImage(q_img)
        self.video_widget.video_label.setPixmap(pixmap)
        self.video_widget.video_label.setAlignment(Qt.AlignCenter)

        # Update the progress bar if it's a video
        if not self.image:
            self.video_prediction_widget.progress_bar.update_time_label(
                self.current_frame
            )

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

    def update_label(self, face, predictions, frame):
        predictions = [item for sublist in predictions for item in sublist]
        if predictions == []:
            predictions = [0, 0, 0, 0, 0]
        x, y, width, height = face.left(), face.top(), face.width(), face.height()
        max_index = predictions.index(max(predictions))
        self.is_fake = 1 if max(predictions) > 0.5 else 0
        label = (
            f"Faked with model: {models_index[max_index]}"
            if self.is_fake
            else "Genuine"
        )
        label_color = (255, 0, 0) if self.is_fake else (0, 255, 0)

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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)
    window = VideoPlayerApp()
    window.show()
    sys.exit(app.exec_())
