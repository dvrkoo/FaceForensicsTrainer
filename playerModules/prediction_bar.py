from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt


class PredictionsBarGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictions = []
        self.models_index = []
        self.placeholder_text = "No predictions/Not selected"
        self.past_predictions = [0]

    def update_width(self):
        total_width = len(self.past_predictions) * (10)
        self.setMinimumWidth((total_width))

    def set_predictions(self, predic, models_indexes, selected_models):
        predictions = predic[:]
        # self.predictions = predic
        self.predictions = [[0]]
        self.models_index = models_indexes
        self.selected_models = selected_models

        self.predictions = predictions.pop(0)
        # self.predictions = predic
        self.past_predictions.append(self.predictions)
        self.update()
        self.update_width()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Check if predictions are available
        if not self.predictions or not self.models_index:
            self.draw_placeholder_text(painter)
            return
        # Define colors for the bars and labels
        # Define font for labels
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        bar_width = self.width()
        # for i, prediction in enumerate(self.predictions):
        # row = i // num_rows  # Row index for the current prediction
        for j in range(len(self.past_predictions)):
            bar_x = int(j * (bar_width + 3) + 5)
            bar_width = 3
            bar_height = -(self.past_predictions[j] * self.height() * 0.5)
            bar_width = int(bar_width)  # Convert to int
            bar_height = int(bar_height)  # Convert to int
            painter.fillRect(
                bar_x,
                int(self.height() * 0.6),
                bar_width,
                bar_height,
                QColor(255, 0, 0)
                if self.past_predictions[j] >= 0.5
                else QColor(0, 255, 0),
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
