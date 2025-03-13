import sys
import os

os.environ.update(
    {
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/home/nick/GitHub/FaceForensicsTrainer/.venv/lib/python3.12/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"
    }
)
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

from playerModules.wavelet_math import batch_packet_preprocessing
from PyQt5.QtWidgets import QMessageBox
from playerModules import model_functions
from playerModules.timer import VideoTimer
from Trufor.src.trufor_test import load_model
from Trufor.visualize import (
    ProcessedImageWidget,
    ProcessedMantraWidget,
    ImageResultsWidget,
)
from playerModules.video_widget import VideoWidget
from playerModules.logo import HomeScreenWidget
from playerModules.mantranet import pre_trained_model, check_forgery
import numpy as np
from playerModules.video_predictions import VideoPredictionWidget
from playerModules.CLIPSynth import predict
from PIL import Image

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


class VideoPlayerApp(QWidget):

    def __init__(self):
        super().__init__()
        self.video_writer = False
        # Set up UI
        self.models = None
        self.detector = None
        self.setup_ui()
        self.freq = False
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
        self.detection_mode = "video"

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

    def check_image_mantra(self, img_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        image = Image.open(img_path)

        # Convert to RGB if it's RGBA or grayscale
        if image.mode != "RGB":
            image = image.convert("RGB")

        MantraNetmodel = pre_trained_model(
            weight_path="trained_models/MantraNetv4.pt", device=device
        )
        figs = check_forgery(MantraNetmodel, img_path=img_path, device=device)
        return figs

    def unload_models(self):
        for i, model in enumerate(self.models):
            self.models[i] = model.cpu()  # Move the model to CPU and update the list
            del self.models[i]  # Delete the model reference to free up memory

        # Optionally, clear the list and force garbage collection
        self.models.clear()
        import gc

        self.current_frame = 0
        self.video_prediction_widget.reset_past_predictions()

        gc.collect()
        torch.cuda.empty_cache()
        print(self.models)
        print("Models unloaded.")
        if not self.freq:
            self.models, self.detector = model_functions.load_freq_models()
            self.video_widget.unload_button.setText("Switch to Pixel Domain")
            self.freq = True
        else:
            self.models, self.detector = model_functions.load_models()
            self.video_widget.unload_button.setText("Switch to Frequency Domain")
            self.freq = False

    def model_selection_changed(self, index):
        # Slot to be triggered when the selected model changes
        selected_model = models_index[index - 1] if index > 0 else None
        self.selected_model = selected_model  # Store the selected model in the class

    def clear_logo_widget(self):
        """Clear the home screen widget."""
        if hasattr(self, "home_widget"):
            self.home_widget.clear()  # Clear the home screen widgets
            self.layout.removeWidget(self.home_widget)
            self.home_widget.deleteLater()
            del self.home_widget  # Optional: delete reference to avoid future errors

    def hide_buttons_for_image(self):
        """Hides the play button and process video button for image detection."""
        self.video_widget.play_button.hide()
        self.video_widget.process_video_button.hide()
        self.video_widget.unload_button.hide()
        self.video_widget.extract_button.hide()
        self.video_widget.image_model_dropdown.show()
        self.video_widget.model_dropdown.hide()

    def show_buttons_for_video(self):
        """Shows the play button and process video button for video detection."""
        self.video_widget.play_button.show()
        self.video_widget.model_dropdown.show()
        self.video_widget.process_video_button.show()
        self.video_widget.unload_button.show()
        self.video_widget.extract_button.show()
        self.video_widget.image_model_dropdown.hide()

    def load_image(self, file_path):
        self.hide_buttons_for_image()
        # let's make a copy rather than a reference
        if hasattr(self, "home_widget"):
            for i in range(
                1, self.video_widget.image_model_dropdown.model().rowCount()
            ):
                item = self.video_widget.image_model_dropdown.model().item(i)
                item2 = self.home_widget.image_model_dropdown.model().item(i)
                if item2.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Checked)
        self.clear_logo_widget()
        if self.processed_image_widget:
            self.processed_image_widget.deleteLater()  # Safely delete the old widget
        self.video_widget.show()
        self.video_widget.toggle_button.setChecked(True)

        # Load the image
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set the image flag and store the image
        self.cap = image_rgb
        self.image = True

        # Hide the video prediction widget if it's an image
        if self.video_prediction_widget:
            self.video_prediction_widget.hide()

        # Get the selected model from the dropdown
        selected_models = self.video_widget.image_model_dropdown.returnSelectedItems()
        if not selected_models:
            print("No model selected in the dropdown.")
            return

        selected_model = selected_models[
            0
        ]  # Assume only one item can be checked at a time
        torch.cuda.empty_cache()
        # Perform actions based on the selected model
        if selected_model == "MantraNet":
            figs = self.check_image_mantra(file_path)
            self.processed_image_widget = ProcessedMantraWidget(figs)
        elif selected_model == "TruFor":
            load_model(file_path)
            self.processed_image_widget = ProcessedImageWidget(file_path)
        elif selected_model == "CLIP_BSID":
            results = predict(file_path)
            self.processed_image_widget = ImageResultsWidget(results)
        self.layout.addWidget(self.processed_image_widget)
        self.display_frame(image_rgb)

    def load_video(self, file_path):
        if self.models is None and self.detector is None:
            self.models, self.detector = model_functions.load_models()
        self.show_buttons_for_video()
        self.video_name = file_path.split("/")[-1]
        self.output_filename = None
        self.video_writer = False
        self.clear_logo_widget()
        self.video_widget.show()

        # Initialize video capture
        self.cap = cv2.VideoCapture(file_path)

        # Get the total number of frames in the video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set the flag to indicate it's not an image
        self.image = False

        # Hide any processed image widget
        if self.processed_image_widget:
            self.processed_image_widget.hide()

        # Show the video prediction widget
        if not self.video_prediction_widget:
            self.setup_video_prediction()
        else:
            self.video_prediction_widget.reset_past_predictions()

        self.video_prediction_widget.show()

        self.video_prediction_widget.progress_bar.set_frame_number(self.total_frames)

        # Start from the first frame
        self.current_frame = 0

        # Read and display the first frame
        if not file_path.endswith((".jpg", ".png")):
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )  # Show the first frame

        # Set the total frame count in the progress bar

        # Show the play button and reset the state
        self.playing = False
        self.video_widget.play_button.show()
        self.video_widget.play_button.setText("Play")

        # Start the video timer (for playback)
        if hasattr(self, "timer"):
            self.timer.start(33)  # ~30 fps for video

    def display_error(self, title, message):
        # Create an error message box
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()  # Show the message box

    def load_media(self):

        print(self.detection_mode)
        # Open a file dialog to choose a video or image file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Video/Image File", "", "Media Files (*.mp4 *.avi *.jpg *.png)"
        )

        if file_path:
            # If video mode is selected, always load as a video
            if self.detection_mode == "video":
                if file_path.endswith((".mp4", ".avi")):
                    self.load_video(file_path)
                else:
                    self.display_error(
                        "Invalid file type",
                        "Please select a video file for video detection.",
                    )
            # If image mode is selected, load based on the file extension
            elif self.detection_mode == "image":
                if file_path.endswith((".jpg", ".png")):
                    self.load_image(file_path)
                else:
                    self.display_error(
                        "Invalid file type",
                        "Please select an image file for image detection.",
                    )

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

        # Initialize VideoWriter if not already done
        if (
            self.video_widget.process_video_button.isChecked()
            and not self.output_filename
        ):
            self.output_filename = self.video_name + f"_{self.current_frame}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(
                self.output_filename, fourcc, fps, (width, height)
            )

        if self.video_widget.process_video_button.isChecked():
            self.video_widget.process_video_button.setText("End Processing")
        if not self.video_widget.process_video_button.isChecked():
            self.video_widget.process_video_button.setText("Process and Save Video")
            self.video_writer = False
            self.output_filename = None

        # Read a frame from the video
        if self.playing and not self.image:
            if self.total_frames != 1:
                self.current_frame += 1
            ret, frame = self.cap.read()
            if ret:
                frame = model_functions.convert_color_space(frame)

                # Perform face detection using dlib
                detected_faces = model_functions.detect_faces(frame, self.detector)
                if detected_faces:
                    # Use the first (largest) detected face
                    face_1 = detected_faces[0]
                    x, y, size = model_functions.get_boundingbox(face_1, frame)
                    face_roi = frame[y : y + size, x : x + size]

                    # Preprocess the face image for model input
                    predictions = []
                    if self.freq:
                        input_tensor = model_functions.preprocess_input_freq(face_roi)
                        input_tensor = input_tensor[np.newaxis, :]
                        input_tensor = batch_packet_preprocessing(input_tensor)
                        input_tensor = torch.from_numpy(input_tensor).to(device)
                    else:
                        input_tensor = model_functions.preprocess_input(face_roi)

                    for model in models_index:
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

                    # Update the frame with predictions
                    self.update_label(face_1, predictions, frame)
                    self.update_predictions(predictions)

                # Write the processed frame to the video writer
                if self.video_writer is not False:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(frame_bgr)

                # Display the processed frame in the UI
                self.display_frame(frame)
            else:
                # End of video
                self.playing = False
                self.video_widget.play_button.setText("Play")
                self.timer.stop()

                # Release the video writer
                if self.video_writer is not False:
                    self.video_writer.release()
                    self.video_writer = False
                    print("Processed video saved as './processed_video_output.mp4'")
        # Image logic (not video playback)
        elif self.image:
            frame = self.cap  # Calculate scaling factors
            self.display_frame(frame)

    def display_frame(self, frame):
        # Reserve a fixed amount of vertical space for the bottom widget
        reserved_space = 100  # adjust this value as needed

        # Use the video label's current dimensions for scaling.
        video_label_height = self.video_widget.video_label.height()
        video_label_width = self.video_widget.video_label.width()

        # Calculate available height by subtracting the reserved space.
        available_height = max(video_label_height - reserved_space, 1)

        # Compute scaling factors for the frame based on the available dimensions.
        height_scale = available_height / frame.shape[0]
        width_scale = video_label_width / frame.shape[1]
        scale = min(height_scale, width_scale)

        # Clamp the scale to ensure it's not too small.
        min_scale = 0.1
        if scale < min_scale:
            scale = min_scale

        # Resize the frame using the computed scale.
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

        # Convert the frame to QImage and then display it.
        q_img = self.opencv_to_qimage(frame_resized)
        pixmap = QPixmap.fromImage(q_img)
        self.video_widget.video_label.setPixmap(pixmap)
        self.video_widget.video_label.setAlignment(Qt.AlignCenter)

        # Update the progress bar if it's a video.
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
