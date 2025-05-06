import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# PyQt imports
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

# Model information for info buttons
MODEL_INFO = {
    "MantraNet": """
<b>MantraNet: Manipulation Tracing Network</b>

MantraNet detects image forgeries by identifying inconsistencies across manipulated regions.

<b>How it works:</b>
- Uses a dual-branch architecture for manipulation trace extraction and anomaly detection
- Analyzes local pixel patterns to detect manipulation artifacts
- Outputs a forgery mask highlighting potential manipulations

<b>Developed by:</b> Yue Wu et al. (2019)
<b>Reference:</b> "ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries"
    """,
    "TruFor": """
<b>TruFor: Trustworthy Forensics Analysis</b>

TruFor is a frequency-aware image forgery detection framework that identifies manipulated regions.

<b>How it works:</b>
- Analyzes both spatial and frequency domain information
- Utilizes wavelet packet transforms to capture multi-scale frequency artifacts
- Produces localization maps highlighting likely manipulated areas
- Generates confidence maps showing detection reliability
- Provides an overall forgery score from 0 (real) to 1 (fake)

<b>Developed by:</b> Davide Cozzolino et al.
<b>Reference:</b> "TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization" 
    """,
    "CLIP_BSID": """
<b>CLIP-BSID: CLIP-Based Synthetic Image Detection</b>

CLIP-BSID leverages CLIP (Contrastive Language-Image Pre-training) to detect AI-generated synthetic images.

<b>How it works:</b>
- Uses two specialized models:
  • CLIPDet: Trained on latent diffusion model outputs (10k+ samples)
  • Corvi: Optimized for modern generative models
- Processes images through CLIP-compatible transformations
- Each model outputs a logit score indicating synthetic probability
- Final score combines individual model outputs using "soft-OR" probability fusion
- Positive scores indicate likely synthetic content; negative for real

<b>Interpretation:</b>
- Higher positive scores indicate stronger confidence in AI generation
- Results display individual model assessments and a final combined score
- Uses semantic understanding to detect inconsistencies invisible to the human eye

<b>Based on:</b> OpenAI's CLIP with specialized synthetic detection capabilities
    """,
}


class BaseVisualizationWidget(QWidget):
    """Base class for visualization widgets with common functionality"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.canvas = None

        # Add info button row
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch(1)
        self.main_layout.addLayout(self.button_layout)

    def add_info_button(self, model_name):
        """Add an info/credits button for the specified model"""
        info_button = QPushButton("ⓘ Credits/Info")
        info_button.setFixedWidth(120)
        info_button.clicked.connect(lambda: self.show_model_info(model_name))
        self.button_layout.addWidget(info_button)

    def show_model_info(self, model_name):
        """Display information about the model in a message box"""
        info_box = QMessageBox()
        info_box.setWindowTitle(f"About {model_name}")
        info_box.setTextFormat(Qt.RichText)
        info_box.setText(
            MODEL_INFO.get(model_name, "Information not available for this model.")
        )
        info_box.setStandardButtons(QMessageBox.Ok)
        info_box.exec_()

    def update_canvas(self, figure):
        """Update the canvas with a new matplotlib figure"""
        # Clear previous canvas if it exists
        if self.canvas:
            self.main_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        # Add new canvas with the figure
        self.canvas = FigureCanvas(figure)
        self.main_layout.addWidget(self.canvas)


class ProcessedMantraWidget(BaseVisualizationWidget):
    """Widget to display MantraNet forgery detection results"""

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.add_info_button("MantraNet")
        self.update_display(results)

    def update_display(self, results):
        """Update the display with MantraNet results"""
        figure = self.create_figure(results)
        self.update_canvas(figure)

    def create_figure(self, results):
        """Create visualization of MantraNet results"""
        figure = plt.figure(figsize=(20, 20))

        # Forgery Mask
        plt.subplot(1, 2, 1)
        plt.imshow(results["forgery_mask"], cmap="gray")
        plt.title("Predicted Forgery Mask")

        # Suspicious Regions
        plt.subplot(1, 2, 2)
        plt.imshow(results["suspicious_regions"])
        plt.title("Suspicious Regions Detected")

        return figure


class ProcessedImageWidget(BaseVisualizationWidget):
    """Widget to display TruFor forgery detection results"""

    def __init__(self, input_image_path, parent=None):
        super().__init__(parent)
        self.add_info_button("TruFor")
        self.update_display(input_image_path)

    def update_display(self, input_image_path):
        """Update the display with TruFor processed results"""
        # Get output path from input image
        result_path = derive_output_path(input_image_path)

        # Create visualization and update canvas
        figure = create_trufor_visualization(result_path)
        self.update_canvas(figure)


class ImageResultsWidget(BaseVisualizationWidget):
    """Widget to display CLIP-BSID synthetic image detection results"""

    # Constants for table configuration
    COLUMNS = ["Model", "Score", "Label"]
    FAKE_COLOR = QColor("red")
    REAL_COLOR = QColor("green")

    def __init__(self, results, parent=None):
        """
        Initialize widget with detection results

        Parameters:
            results (dict): Dictionary with model names as keys and scores as values
                           (typically contains 'CLIPDet', 'Corvi', and 'Final Score')
        """
        super().__init__(parent)
        self.add_info_button("CLIP_BSID")
        self.setWindowTitle("Synthetic Image Detection Results")

        # Create and configure the results table
        results_table = self.create_results_table(results)
        self.main_layout.addWidget(results_table)

    def create_results_table(self, results):
        """Create a formatted table displaying model results"""
        # Create table with headers
        table = QTableWidget(len(results), len(self.COLUMNS))
        table.setHorizontalHeaderLabels(self.COLUMNS)

        # Apply styling
        self.apply_table_styling(table)

        # Populate table with data
        for row_idx, (model_name, score) in enumerate(results.items()):
            self._add_result_row(table, row_idx, model_name, score)

        return table

    def _add_result_row(self, table, row_idx, model_name, score):
        """Add a single result row to the table"""
        # Create items
        model_item = QTableWidgetItem(model_name)
        score_item = QTableWidgetItem(f"{score:.4f}")

        # Determine if the image is fake (positive score indicates synthetic)
        is_fake = score > 0
        label_text = "Synthetic" if is_fake else "Real"
        result_color = self.FAKE_COLOR if is_fake else self.REAL_COLOR

        # Create label item
        label_item = QTableWidgetItem(label_text)

        # Apply center alignment to all items
        for item in [model_item, score_item, label_item]:
            item.setTextAlignment(Qt.AlignCenter)

        # Apply color to score and label
        score_item.setForeground(result_color)
        label_item.setForeground(result_color)

        # Add items to table
        table.setItem(row_idx, 0, model_item)
        table.setItem(row_idx, 1, score_item)
        table.setItem(row_idx, 2, label_item)

    def apply_table_styling(self, table):
        """Apply consistent styling to results table"""
        # Set table style
        table.setStyleSheet("""
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
        """)

        # Configure table appearance
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


def derive_output_path(input_image_path):
    """Get the corresponding output .npz file path for a given input image"""
    # Extract base name and extension
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    extension = os.path.splitext(os.path.basename(input_image_path))[1]

    # Construct output path
    return os.path.join("./Trufor/output", f"{base_name}{extension}.npz")


def create_trufor_visualization(result_path):
    """Create a visualization of TruFor results"""
    # Load result data
    result_data = np.load(result_path)

    # Create figure with subplots
    fig = Figure(figsize=(8, 6))
    axes = fig.subplots(1, 2)

    # Set title with score
    fig.suptitle(f"Score: {result_data['score']:.3f}")

    # Draw localization map
    axes[0].imshow(result_data["map"], cmap="RdBu_r", clim=[0, 1])
    axes[0].set_title("Localization map")
    axes[0].axis("off")

    # Draw confidence map
    axes[1].imshow(result_data["conf"], cmap="gray", clim=[0, 1])
    axes[1].set_title("Confidence map")
    axes[1].axis("off")

    return fig

