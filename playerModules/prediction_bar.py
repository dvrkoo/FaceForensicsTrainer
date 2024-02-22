from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt


class PredictionsBarGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictions = []
        self.models_index = []
        self.placeholder_text = "No predictions"

    def set_predictions(self, predictions, models_index):
        self.predictions = predictions
        self.models_index = models_index
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Check if predictions are available
        if not self.predictions or not self.models_index:
            self.draw_placeholder_text(painter)
            return

        # Define colors for the bars and labels
        fake_color = QColor(255, 0, 0)
        real_color = QColor(0, 255, 0)
        text_color = Qt.black

        # Define font for labels
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)

        bar_height = self.height() / len(self.predictions)
        max_prediction = max(self.predictions)

        for i, prediction in enumerate(self.predictions):
            bar_width = prediction / max_prediction * self.width()
            bar_x = 0  # Convert to int
            bar_y = int(bar_height * i)  # Convert to int
            bar_width = int(bar_width)  # Convert to int
            bar_height = int(bar_height)  # Convert to int
            bar_color = fake_color if prediction >= 0.5 else real_color
            painter.fillRect(bar_x, bar_y, bar_width, bar_height, bar_color)

            # Display model name and prediction value as labels
            label_text = f"{self.models_index[i]}: {prediction:.4f}"
            painter.setPen(text_color)
            text_color = Qt.white
            font = QFont()
            font.setPointSize(12)  # Adjust font size as needed
            painter.setFont(font)
            painter.setPen(text_color)

            # Calculate center coordinates of the widget
            center_x = self.width() / 2

            # Get the size of the text
            text_rect = painter.fontMetrics().boundingRect(label_text)

            # Draw the text, adjusting the coordinates to center it
            painter.drawText(
                int(center_x - text_rect.width() / 2),
                bar_y + 25,
                label_text,
            )

    def draw_placeholder_text(self, painter):
        text_color = Qt.black

        # Define font for placeholder text
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)

        # Draw placeholder text in the center of the widget
        painter.setPen(text_color)
        painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)
