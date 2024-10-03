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
    QSizePolicy,
    QScrollArea,
    QGridLayout,
)


from playerModules import model_functions, prediction_bar
from playerModules.models_dropdown import CheckableComboBox
from playerModules.progress_bar import ProgressBarWithTimeLabel
from playerModules.timer import VideoTimer
from Trufor.src.trufor_test import load_model
from Trufor.visualize import ProcessedImageWidget

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
        self.progress_bar = None
        self.bottom_layout = None

    def setup_trufor():
        pass

    def model_selection_changed(self, index):
        # Slot to be triggered when the selected model changes
        selected_model = models_index[index - 1] if index > 0 else None
        self.selected_model = selected_model  # Store the selected model in the class

    # Usage example:

    # In your PyQt application, you would call this function where image_np is the loaded image.
    def setup_ui(self):
        # Set up layout
        self.faceswap = None
        self.processed_widget = None
        self.loading_label = QLabel("Loading Image...", self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.hide()  # Initially hidden
        self.bottom_flag = False
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.loading_label)
        video_layout = QHBoxLayout()
        video_layout.setSpacing(0)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addLayout(video_layout, stretch=1)
        buttons_layout = QVBoxLayout()
        video_layout.addLayout(buttons_layout)
        self.model_dropdown = CheckableComboBox()
        self.model_dropdown.addItems(models_index)
        self.video_label = QLabel(self)
        # self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignLeft)
        buttons_layout.addWidget(self.model_dropdown)
        video_layout.addWidget(self.video_label)
        buttons_layout.setAlignment(Qt.AlignCenter)
        # Add play and pause buttons
        self.play_button = QPushButton("Pause", self)
        self.play_button.clicked.connect(self.play_pause_video)
        self.play_button.hide()
        buttons_layout.addWidget(self.play_button)
        # Add a button to open a file dialog
        self.load_button = QPushButton("Load Video/Image", self)
        self.load_button.clicked.connect(self.load_video)
        buttons_layout.addWidget(self.load_button)
        # self.setup_video_prediction()

    def setup_video_prediction(self):
        if not self.progress_bar and not self.bottom_layout:
            self.progress_bar = ProgressBarWithTimeLabel(self)
            self.layout.addWidget(self.progress_bar)

            self.bottom_widget = QWidget()
            self.bottom_layout = QHBoxLayout(self.bottom_widget)
            # self.layout.addLayout(self.bottom_layout, stretch=1)
            self.layout.addWidget(self.bottom_widget, stretch=1)

            self.scroll_area = QScrollArea(self)
            self.scroll_area.setWidgetResizable(True)  # Allow prediction_bar to expand
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Add

        self.setup_predictions_text()

    def setup_image_prediction(self):
        self.bottom_flag = False
        if not self.progress_bar and not self.bottom_layout:
            self.progress_bar = ProgressBarWithTimeLabel(self)
            self.layout.addWidget(self.progress_bar)
            self.bottom_layout = QHBoxLayout()
            self.layout.addLayout(self.bottom_layout, stretch=1)

            self.scroll_area = QScrollArea(self)
            self.scroll_area.setWidgetResizable(True)  # Allow prediction_bar to expand
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Add

    def setup_predictions_text(self):
        if not self.faceswap:
            self.bottom_flag = True
            self.text_widget = QGridLayout()
            self.faceswap = QLabel("Faceswap : ")
            self.faceswap.setFixedWidth(150)
            self.text_widget.addWidget(self.faceswap, 0, 0, 1, 1)
            self.deepfake = QLabel("Deepfake : ")
            self.text_widget.addWidget(self.deepfake, 1, 0, 1, 1)
            self.neuraltextures = QLabel("Neuraltextures : ")
            self.text_widget.addWidget(self.neuraltextures, 2, 0, 1, 1)
            self.face2face = QLabel("Face2Face : ")
            self.text_widget.addWidget(self.face2face, 3, 0, 1, 1)
            self.faceshift = QLabel("Faceshifter : ")
            self.text_widget.addWidget(self.faceshift, 4, 0, 1, 1)
            self.credits = QPushButton("Credits")
            self.text_widget.addWidget(self.credits, 5, 0, 1, 1)
            self.bottom_layout.addLayout(self.text_widget)
            self.bottom_layout.addWidget(self.scroll_area)
        self.setup_predictions_bars()

    def setup_predictions_bars(self):
        self.predictions_widget = QWidget()
        self.predictions_layout = QGridLayout(self.predictions_widget)
        self.faceswap_bar = prediction_bar.PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.faceswap_bar, 0, 1, 1, 1)
        self.deepfake_bar = prediction_bar.PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.deepfake_bar, 1, 1, 1, 1)
        self.neuraltextures_bar = prediction_bar.PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.neuraltextures_bar, 2, 1, 1, 1)
        self.face2face_bar = prediction_bar.PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.face2face_bar, 3, 1, 1, 1)
        self.faceshift_bar = prediction_bar.PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.faceshift_bar, 4, 1, 1, 1)
        self.scroll_area.setWidget(self.predictions_widget)

    def update_predictions_texts(self, predictions):
        self.faceswap.setText(f"Faceswap : {predictions[0][0]:.4f}% Fake")
        self.deepfake.setText(f"Deepfake : {predictions[1][0]:.4f}% Fake")
        self.neuraltextures.setText(f"Neuraltextures : {predictions[2][0]:.4f}% Fake")
        self.face2face.setText(f"Face2Face : {predictions[3][0]:.4f}% Fake")
        self.faceshift.setText(f"Faceshifter : {predictions[4][0]:.4f}% Fake")

    def clear_layout(self, layout):
        """Utility function to clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)  # Get the first item
            if item.widget():  # If it's a widget
                item.widget().deleteLater()  # Schedule for deletion
            del item  # Remove reference to item

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
            # If a file is selected, initialize video capture
            self.cap = cv2.VideoCapture(file_path)
            # Get the total number of frames in the video (1 for an image)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Check if it's an image by looking at file extension
            if file_path.endswith((".jpg", ".png")):
                if self.progress_bar and self.bottom_layout:
                    self.progress_bar.hide()
                    self.bottom_widget.hide()

                if self.processed_widget:
                    self.layout.removeWidget(self.processed_widget)
                    self.processed_widget.deleteLater()  # Safely delete the old widget

                image = cv2.imread(file_path)
                print(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.cap = image_rgb
                self.image = True
                # self.setup_image_prediction()
                self.loading_label.show()
                load_model(file_path)
                self.loading_label.hide()
                self.processed_widget = ProcessedImageWidget(file_path)
                self.layout.addWidget(self.processed_widget)
            else:
                if self.processed_widget:
                    self.processed_widget.hide()
                self.setup_video_prediction()
                self.progress_bar.show()
                self.bottom_widget.show()
                self.image = False
                # self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Show the play button and reset the state
            self.current_frame = 0
            self.playing = False
            self.play_button.show()
            self.play_button.setText("Play")
            # Start the video timer
            if hasattr(self, "timer"):
                self.timer.start(33)  # ~30 fps for video
            self.load_button.hide()
            if self.bottom_flag and not self.image:
                self.progress_bar.set_frame_number(self.total_frames)

    def display_frame(self, frame):
        """Helper function to display an image or video frame in the QLabel."""
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
        self.faceswap_bar.set_predictions(
            predictions[0], models_index, self.selected_models
        )
        self.deepfake_bar.set_predictions(
            predictions[1], models_index, self.selected_models
        )
        self.neuraltextures_bar.set_predictions(
            predictions[2], models_index, selected_model
        )
        self.face2face_bar.set_predictions(predictions[3], models_index, selected_model)
        self.faceshift_bar.set_predictions(predictions[4], models_index, selected_model)
        self.update_predictions_texts(predictions)

    def timerEvent(self):
        self.selected_models = self.model_dropdown.returnSelectedItems()
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
                    self.update_predictions(predictions, self.selected_models)
                    self.update_predictions_texts(predictions)
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
                self.progress_bar.update_time_label(self.current_frame)

            else:
                self.playing = False
                self.play_button.setText("Play")
                self.load_button.show()
                self.timer.stop()
        # frame logic
        elif self.image:
            frame = self.cap  # Calculate scaling factors
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
            # self.progress_bar.update_time_label(self.current_frame)

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
