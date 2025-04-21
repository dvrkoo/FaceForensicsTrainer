import sys
import os

os.environ.update(
    {
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/home/nick/GitHub/FaceForensicsTrainer/.venv/lib/python3.12/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"
    }
)
import cv2
import torch
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtWidgets import QSizePolicy
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

MAX_FRAME_WIDTH = 800
MAX_FRAME_HEIGHT = 600


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
        self.container_layout.addWidget(self.home_widget)
        self.video_prediction_widget = None
        self.detection_mode = "video"

    def setup_ui(self):
        self.processed_image_widget = None
        # Main layout for the entire widget
        self.layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.container_widget = QWidget()
        self.container_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.container_layout = QVBoxLayout(self.container_widget)
        self.video_widget = VideoWidget(self)
        self.video_widget.hide()  # Initially hidden
        self.container_layout.addWidget(self.video_widget)
        self.scroll_area.setWidget(self.container_widget)
        self.layout.addWidget(self.scroll_area)

    def setup_video_prediction(self):
        """Initializes the video prediction widget and adds it to the main layout."""
        self.video_prediction_widget = VideoPredictionWidget(self)
        self.container_layout.addWidget(self.video_prediction_widget)

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
            # Show an error message and return
            self.display_error(
                "No model selected",
                "Please select a model from the dropdown to proceed.",
            )
            return

        selected_model = selected_models[
            0
        ]  # Assume only one item can be checked at a time
        torch.cuda.empty_cache()
        # Perform actions based on the selected model
        if selected_model == "MantraNet":
            try:
                figs = self.check_image_mantra(file_path)
                self.processed_image_widget = ProcessedMantraWidget(figs)
            except Exception as e:
                print(e)
                self.display_error(
                    "Error",
                    "An error occurred while processing the image. Most likely, the image is too large to fit in memory.",
                )
        elif selected_model == "TruFor":
            try:
                self.models, self.detector = model_functions.load_freq_models()
                self.processed_image_widget = ProcessedImageWidget(file_path)
            except Exception as e:
                print(e)
                self.display_error(
                    "Error",
                    "An error occurred while processing the image. Most likely, the image is too large to fit in memory.",
                )
        elif selected_model == "CLIP_BSID":
            try:
                results = predict(file_path)
                self.processed_image_widget = ImageResultsWidget(results)
            except Exception as e:
                print(e)
                self.display_error(
                    "Error",
                    "An error occurred while processing the image. Most likely, the image is too large to fit in memory.",
                )
        self.container_layout.addWidget(self.processed_image_widget)
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
        # Resize frame to fit within the maximum size
        frame_resized = self.resize_frame(frame, MAX_FRAME_WIDTH, MAX_FRAME_HEIGHT)

        # Convert frame to QImage and display it
        q_img = self.opencv_to_qimage(frame_resized)
        pixmap = QPixmap.fromImage(q_img)
        self.video_widget.video_label.setPixmap(pixmap)
        self.video_widget.video_label.setAlignment(Qt.AlignCenter)

        # Update progress bar if displaying a video
        if not self.image:
            self.video_prediction_widget.progress_bar.update_time_label(
                self.current_frame
            )

    def resize_frame(self, frame, max_width, max_height):
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)  # Ensure it never scales up
        return cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
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


def apply_stylesheet():
    """Apply a custom Catppuccin Latte stylesheet to the application."""
    QApplication.setStyle("Fusion")

    # Enable high-DPI scaling attributes
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    # Force integer scaling on macOS
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Round"

    # Catppuccin Latte color palette (light theme)
    base = "#eff1f5"  # Background
    mantle = "#e6e9ef"  # Slightly darker background
    crust = "#dce0e8"  # Border color
    text = "#4c4f69"  # Text color
    subtext = "#5c5f77"  # Dimmed text
    surface0 = "#ccd0da"  # Surface
    surface1 = "#bcc0cc"  # Higher surface
    blue = "#1e66f5"  # Blue
    lavender = "#7287fd"  # Lavender
    mauve = "#8839ef"  # Purple
    green = "#40a02b"  # Green
    peach = "#fe640b"  # Peach

    stylesheet = f"""
    QWidget {{
        background-color: {base};
        color: {text};
        font-family: '-apple-system', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
        font-size: 14px;
    }}
    
    QPushButton {{
        background-color: {surface0};
        border: 1px solid {surface1};
        border-radius: 4px;
        padding: 6px 14px;
        color: {text};
        font-weight: normal;
        min-height: 28px;
    }}
    
    QPushButton:hover {{
        background-color: {surface1};
        border-color: {lavender};
    }}
    
    QPushButton:pressed {{
        background-color: {crust};
        border-color: {blue};
    }}
    
    QPushButton:checked {{
        background-color: {lavender};
        color: {base};
    }}
    
    QLabel {{
        color: {text};
    }}
    
    QProgressBar {{
        border: 1px solid {surface1};
        border-radius: 4px;
        text-align: center;
        background-color: {surface0};
        color: {text};
    }}
    
    QProgressBar::chunk {{
        background-color: {blue};
        border-radius: 2px;
    }}
    
    QScrollBar {{
        background-color: {surface0};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle {{
        background-color: {surface1};
        border-radius: 6px;
        min-height: 30px;
    }}
    
    QScrollBar::handle:hover {{
        background-color: {lavender};
    }}
    
    QComboBox {{
        border: 1px solid {surface1};
        border-radius: 4px;
        padding: 4px;
        background-color: {mantle};
        color: {text};
        min-height: 28px;
    }}
    
    QComboBox::drop-down {{
        border: 0px;
        background-color: {surface0};
        width: 24px;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }}
    
    QCheckBox {{
        color: {text};
        spacing: 5px;
    }}
    
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {surface1};
        border-radius: 3px;
        background-color: {mantle};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {lavender};
    }}
    
    QGroupBox {{
        border: 1px solid {surface1};
        border-radius: 4px;
        margin-top: 0.5em;
        padding-top: 0.5em;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px;
        color: {subtext};
    }}
        QScrollBar:vertical {{ /* Explicitly target vertical */
        background-color: {mantle}; /* Use a slightly darker background for track: #e6e9ef */
        width: 14px;              /* Maybe slightly wider */
        border-radius: 7px;
        border: 1px solid {crust}; /* Add a subtle border: #dce0e8 */
        margin: 1px; /* Add tiny margin */
    }}
    QScrollBar::handle:vertical {{ /* Explicitly target vertical handle */
        background-color: {subtext}; /* Use a much darker handle: #5c5f77 */
        min-height: 30px;
        border-radius: 6px;
        border: 1px solid {text};  /* Add handle border: #4c4f69 */
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {blue}; /* Keep hover distinct: #1e66f5 */
        border-color: {blue};
    }}

    /* Add equivalent styles if you might have horizontal scrollbars */
    QScrollBar:horizontal {{
        background-color: {mantle};
        height: 14px;
        border-radius: 7px;
        border: 1px solid {crust};
        margin: 1px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {subtext};
        min-width: 30px;
        border-radius: 6px;
        border: 1px solid {text};
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: {blue};
        border-color: {blue};
    }}

    /* Hide the default arrow buttons if you don't need them */
    QScrollBar::add-line, QScrollBar::sub-line {{
        background: none;
        border: none;
        height: 0px;
        width: 0px;
    }}
    /* Style the page areas (track area not covered by handle) */
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: none; /* Make page area transparent */
    }}
    """

    QApplication.instance().setStyleSheet(stylesheet)


if __name__ == "__main__":
    # Set environment variables first
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    # Set up the QApplication
    app = QApplication(sys.argv)

    # Apply stylesheet after creating the application
    apply_stylesheet()

    # Create and show the main window
    window = VideoPlayerApp()
    window.show()

    # Start the application
    sys.exit(app.exec_())
