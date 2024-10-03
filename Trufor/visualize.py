import os
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFileDialog, QLabel
import cv2
from PyQt5.QtWidgets import QMainWindow


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
    fig.suptitle(
        "Score: %.3f - %s"
        % (result["score"], "Fake" if result["score"] > 0.5 else "Not Fake")
    )

    axs[0].imshow(result["map"], cmap="RdBu_r", clim=[0, 1])
    axs[0].set_title("Localization map")
    axs[0].axis("off")  # Hide axis for better visualization

    axs[1].imshow(result["conf"], cmap="gray", clim=[0, 1])
    axs[1].set_title("Confidence map")
    axs[1].axis("off")  # Hide axis for better visualization

    # Return the figure to be embedded in a PyQt widget
    return fig


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
