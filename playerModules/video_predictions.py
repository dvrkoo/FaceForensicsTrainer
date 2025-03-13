from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QPushButton,
    QScrollArea,
    QHBoxLayout,
)
from .progress_bar import ProgressBarWithTimeLabel
from .prediction_bar import PredictionsBarGraph

models_index = ["faceswap", "deepfake", "neuraltextures", "face2face", "faceshifter"]


class VideoPredictionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up progress bar and bottom layout
        self.progress_bar = None
        self.bottom_layout = None
        self.text_widget = None
        self.scroll_area = None
        self.bottom_widget = None
        self.bottom_flag = False

        # Predictions labels and bars
        self.faceswap = None
        self.deepfake = None
        self.neuraltextures = None
        self.face2face = None
        self.faceshift = None
        self.credits = None
        self.predictions_widget = None
        self.setMinimumHeight(500)

        self.setup_ui()

    def setup_ui(self):
        """Sets up the main UI for video prediction."""
        # Initialize progress bar and bottom layout if not already created
        if not self.progress_bar and not self.bottom_layout:
            self.progress_bar = ProgressBarWithTimeLabel(self)
            self.layout = QVBoxLayout(self)
            self.layout.addWidget(self.progress_bar)

            self.bottom_widget = QWidget()
            self.bottom_layout = QHBoxLayout(self.bottom_widget)
            self.layout.addWidget(self.bottom_widget)

            self.scroll_area = QScrollArea(self)
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Set up predictions texts and bars
        self.setup_predictions_text()

    def setup_predictions_text(self):
        """Sets up the labels for different prediction categories."""
        if not self.faceswap:
            self.bottom_flag = True
            self.text_widget = QGridLayout()

            # Create labels for prediction categories
            self.faceswap = QLabel("Faceswap : ")
            self.faceswap.setFixedWidth(200)
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
        """Sets up the bar graphs for each prediction category."""
        self.predictions_widget = QWidget()
        self.predictions_layout = QGridLayout(self.predictions_widget)

        # Create bar graphs for each prediction
        self.faceswap_bar = PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.faceswap_bar, 0, 1, 1, 1)

        self.deepfake_bar = PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.deepfake_bar, 1, 1, 1, 1)

        self.neuraltextures_bar = PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.neuraltextures_bar, 2, 1, 1, 1)

        self.face2face_bar = PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.face2face_bar, 3, 1, 1, 1)

        self.faceshift_bar = PredictionsBarGraph(self)
        self.predictions_layout.addWidget(self.faceshift_bar, 4, 1, 1, 1)

        self.scroll_area.setWidget(self.predictions_widget)

    def update_predictions_texts(self, predictions):
        """Updates the prediction texts based on new data."""
        self.faceswap.setText(f"Faceswap : {predictions[0][0]*100:.4f}% Fake")
        self.deepfake.setText(f"Deepfake : {predictions[1][0]*100:.4f}% Fake")
        self.neuraltextures.setText(
            f"Neuraltextures : {predictions[2][0]*100:.4f}% Fake"
        )
        self.face2face.setText(f"Face2Face : {predictions[3][0]*100:.4f}% Fake")
        self.faceshift.setText(f"Faceshifter : {predictions[4][0]*100:.4f}% Fake")

    def update_predictions(self, predictions, selected_models):
        self.faceswap_bar.set_predictions(predictions[0], models_index, selected_models)
        self.deepfake_bar.set_predictions(predictions[1], models_index, selected_models)
        self.neuraltextures_bar.set_predictions(
            predictions[2], models_index, selected_models
        )
        self.face2face_bar.set_predictions(
            predictions[3], models_index, selected_models
        )
        self.faceshift_bar.set_predictions(
            predictions[4], models_index, selected_models
        )
        self.update_predictions_texts(predictions)

    def reset_past_predictions(self):
        self.faceswap_bar.past_predictions = [0]
        self.deepfake_bar.past_predictions = [0]
        self.neuraltextures_bar.past_predictions = [0]
        self.face2face_bar.past_predictions = [0]
        self.faceshift_bar.past_predictions = [0]
