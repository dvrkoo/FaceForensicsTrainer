import os
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFileDialog, QLabel
import cv2
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtGui import QColor


class ProcessedMantraWidget(QWidget):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = None  # Initially, there is no canvas

        self.update_display(results)

    def update_display(self, results):
        """Update the display with the processed image based on the input image path."""
        # Clear the previous canvas if it exists
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()  # Safely remove the previous canvas

        # Process the image and get the results

        # Create a Matplotlib figure to display the results
        figure = self.create_figure(results)

        # Embed the Matplotlib figure in the widget
        self.canvas = FigureCanvas(figure)
        self.layout.addWidget(self.canvas)

    def create_figure(self, results):
        """Create a Matplotlib figure to display the processed results."""
        figure = plt.figure(figsize=(20, 20))

        # Original Image

        # Forgery Mask
        plt.subplot(1, 2, 1)
        plt.imshow(results["forgery_mask"], cmap="gray")
        plt.title("Predicted Forgery Mask")

        # Suspicious Regions
        plt.subplot(1, 2, 2)
        plt.imshow(results["suspicious_regions"])
        plt.title("Suspicious Regions Detected")

        return figure


def derive_output_path(input_image_path):
    """Derive the corresponding output .npz file path from the input image path."""
    # Extract the base name of the input image
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    extension = os.path.splitext(os.path.basename(input_image_path))[1]

    # Construct the corresponding output path with .npz extension
    output_path = os.path.join("./Trufor/output", f"{base_name}{extension}.npz")

    return output_path


def display_processed_image(result_path):
    """Create a Matplotlib figure displaying processed image results."""
    result = np.load(result_path)

    cols = 2
    # Create a PyQt figure widget to show the results
    fig = Figure(figsize=(8, 6))
    axs = fig.subplots(1, cols)

    # Set the title based on the score
    fig.suptitle("Score: %.3f " % result["score"])
    # fig.suptitle(
    #     "Score: %.3f - %s"
    #     % (result["score"], "Fake" if result["score"] > 0.5 else "Not Fake")
    # )

    axs[0].imshow(result["map"], cmap="RdBu_r", clim=[0, 1])
    axs[0].set_title("Localization map")
    axs[0].axis("off")  # Hide axis for better visualization

    axs[1].imshow(result["conf"], cmap="gray", clim=[0, 1])
    axs[1].set_title("Confidence map")
    axs[1].axis("off")  # Hide axis for better visualization

    # Return the figure to be embedded in a PyQt widget
    return fig


class ImageResultsWidget(QWidget):
    def __init__(self, results, parent=None):
        """
        Parameters:
            results (dict): A dictionary with keys as model names and values as scores.
        """
        super(ImageResultsWidget, self).__init__(parent)
        self.setWindowTitle("Fusion Results")
        layout = QVBoxLayout(self)

        # Create a table widget with three columns: Model, Score, Label
        table = QTableWidget(len(results), 3)
        table.setHorizontalHeaderLabels(["Model", "Score", "Label"])

        # Apply styling using a stylesheet
        table.setStyleSheet(
            """
            QTableWidget {
                background-color: #f9f9f9;
                gridline-color: #dcdcdc;
                font-size: 18px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 6px;
                border: 1px solid #dcdcdc;
                font-weight: bold;
            }
        """
        )
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Populate the table
        for i, (model, score) in enumerate(results.items()):
            model_item = QTableWidgetItem(model)
            model_item.setTextAlignment(Qt.AlignCenter)

            score_item = QTableWidgetItem(f"{score:.4f}")
            score_item.setTextAlignment(Qt.AlignCenter)

            # Determine label and color based on the score
            if score > 0:
                label_text = "Fake"
                color = QColor("red")
            else:
                label_text = "Real"
                color = QColor("green")
            label_item = QTableWidgetItem(label_text)
            label_item.setTextAlignment(Qt.AlignCenter)

            # Set the text color for score and label
            score_item.setForeground(color)
            label_item.setForeground(color)

            table.setItem(i, 0, model_item)
            table.setItem(i, 1, score_item)
            table.setItem(i, 2, label_item)

        layout.addWidget(table)
        self.setLayout(layout)


class ProcessedImageWidget(QWidget):
    def __init__(self, input_image_path, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = None  # Initially, there is no canvas

        self.update_display(input_image_path)

    def update_display(self, input_image_path):
        """Update the display with the processed image based on the input image path."""
        # Derive the output image path from the input image path
        result_path = derive_output_path(input_image_path)

        # Clear the previous canvas if it exists
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()  # Safely remove the previous canvas

        # Display the processed image in the widget
        figure = display_processed_image(result_path)
        self.canvas = FigureCanvas(figure)

        # Add the canvas to the widget layout
        self.layout.addWidget(self.canvas)


class VideoPlayerApp(QMainWindow):  # Assuming your class is named VideoPlayerApp
    def __init__(self):
        super().__init__()
        # Your initialization code here
        self.layout = QVBoxLayout()  # Example layout, adjust as needed
        self.processed_widget = None  # Initialize your processed widget

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

            # Check if it's an image by looking at file extension
            if file_path.endswith((".jpg", ".png")):
                self.image = True
                if self.processed_widget is None:
                    self.processed_widget = ProcessedImageWidget(file_path)
                    self.layout.addWidget(self.processed_widget)
                else:
                    self.processed_widget.update_display(file_path)
            else:
                # Handle video loading (you can implement video handling logic here)
                if self.processed_widget:
                    self.processed_widget.hide()  # Hide processed widget for video

            # Additional setup and UI updates...
            self.play_button.show()
            self.play_button.setText("Play")
