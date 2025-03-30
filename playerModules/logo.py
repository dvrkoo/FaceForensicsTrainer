from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from .models_dropdown import CheckableComboBox

image_models = ["TruFor", "MantraNet", "CLIP_BSID"]


class LogoPlaceholderWidget(QWidget):
    def __init__(self, logo_paths, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        for logo_path in logo_paths:
            label = QLabel(self)
            pixmap = QPixmap(logo_path)
            label.setPixmap(pixmap)
            # label.setScaledContents(True)  # Scale the pixmap to fit the label
            label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(label)


class HomeScreenWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = parent
        # Set up the layout for the home screen
        self.layout = QVBoxLayout(self)
        # Create and configure the instructions label
        self.instructions_label = QLabel(self)
        self.instructions_label.setText(
            "Welcome to the DeepFake Detection Application!\n\n"
            "This application analyzes images and videos to determine whether they are real or manipulated. "
            "To get started, please load a video or image file using the 'Load Video' button. "
            "If you load an Image will see the results along with confidence maps indicating the likelihood of manipulation."
            "If you load a video, you will see the video playing along with the real-time predictions of 5 particular forgeries."
        )
        self.instructions_label.setAlignment(Qt.AlignCenter)
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setFont(QFont("Arial", 16))
        self.instructions_label.setStyleSheet("padding: 10px; color: #333;")

        # Logo images and widget
        logo_paths = [
            "./img/micc_logo-1-754x380.png",
            "./img/Logo_Blu_Trasparente-sc.png",
        ]
        self.logo_widget = LogoPlaceholderWidget(logo_paths)

        # Load button
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(
            parent.load_media
        )  # Connect to parent's load_video method

        # Add widgets to layout
        self.layout.addWidget(self.instructions_label)
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.logo_widget)
        self.image_model_dropdown = CheckableComboBox(image=True)
        self.image_model_dropdown.addItems(image_models)
        self.layout.addWidget(self.image_model_dropdown)
        self.image_model_dropdown.hide()

        # Add this in your setup method where you define the UI elements
        self.toggle_button = QPushButton("Switch to Image Detection", self)
        self.toggle_button.setCheckable(True)  # This makes it act as a toggle button
        self.toggle_button.setChecked(False)  # Initially set to image detection mode
        self.toggle_button.clicked.connect(self.switch_detection_mode)
        self.layout.addWidget(self.toggle_button)

    def switch_detection_mode(self):
        # Toggle between image and video detection mode
        if self.toggle_button.isChecked():
            self.toggle_button.setText("Switch to Video Detection")
            self.load_button.setText("Load Image")
            self.player.detection_mode = "image"
            print(self.player.detection_mode)
            self.image_model_dropdown.show()
        else:
            self.image_model_dropdown.hide()
            self.toggle_button.setText("Switch to Image Detection")
            self.load_button.setText("Load Video")
            self.player.detection_mode = "video"

    def clear(self):
        """Clear all widgets in the home screen layout."""
        self.layout.removeWidget(self.instructions_label)
        self.instructions_label.deleteLater()  # Delete the widget

        self.layout.removeWidget(self.load_button)
        self.load_button.deleteLater()

        self.layout.removeWidget(self.logo_widget)
        self.logo_widget.deleteLater()

        # Optionally clear references
        del self.instructions_label
        del self.load_button
        del self.logo_widget
