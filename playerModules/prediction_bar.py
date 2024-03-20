from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt


class PredictionsBarGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictions = []
        self.models_index = []
        self.placeholder_text = "No predictions"
        # self.setMinimumSize(600, 500)
        self.total_spaces = 5
        self.spacing = 10
        print(self.height())
        self.space_height = self.update_space_height()
        self.past_predictions = [[] for _ in range(5)]

    def update_width(self):
        total_width = len(self.past_predictions[0]) * (10)
        self.setMinimumWidth(total_width)

    def set_predictions(self, predic, models_indexes, selected_models):
        predictions = predic[:]
        # self.predictions = predic
        self.predictions = [[0] for i in range(5)]
        self.models_index = models_indexes
        self.selected_models = selected_models
        for selected_model in self.selected_models:
            for i in range(5):
                if i == self.models_index.index(selected_model):
                    self.predictions[i] = predictions.pop(0)
        # self.predictions = predic
        for i in range(5):
            self.past_predictions[i].append(self.predictions[i][0])
        self.update()
        self.update_width()

    def update_space_height(self):
        self.space_height = (self.height() - (self.total_spaces - 1) * self.spacing) / (
            self.total_spaces + 1
        )

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
        bar_width = self.width() / len(self.predictions)
        num_rows = (
            len(self.predictions) + self.total_spaces - 1
        ) // self.total_spaces  # Number of columns needed
        for i, prediction in enumerate(self.predictions):
            row = i // num_rows  # Row index for the current prediction
            for j in range(len(self.past_predictions[i])):
                bar_x = int(j * (bar_width + 3) + 5)
                bar_y = int(
                    70 + (row * self.space_height) // num_rows + row * self.spacing
                )
                bar_width = 3
                bar_height = -(self.past_predictions[i][j] * self.space_height * 0.5)
                bar_width = int(bar_width)  # Convert to int
                bar_height = int(bar_height)  # Convert to int
                painter.fillRect(
                    bar_x,
                    bar_y,
                    bar_width,
                    bar_height,
                    QColor(255, 0, 0)
                    if self.past_predictions[i][j] >= 0.5
                    else QColor(0, 255, 0),
                )
            #     # Display model name and prediction value as labels
            # label_text = f"{self.models_index[i]}: {prediction[0]:.4f}"
            # painter.setPen(text_color)
            # text_color = Qt.white
            # font = QFont()
            # font.setPointSize(12)  # Adjust font size as needed
            # painter.setFont(font)
            # painter.setPen(text_color)
            # # Draw the text, adjusting the coordinates to center it
            # painter.drawText(
            #     0,
            #     bar_y + 25,
            #     label_text,

    def return_bar_y(self):
        bars_y = []
        num_rows = 5
        tmp = 0
        for row in range(5):
            if self.space_height:
                tmp = int(
                    70 + (row * self.space_height) // num_rows + row * self.spacing
                )
            else:
                tmp = 0
            bars_y.append(tmp)
        return bars_y

    def resizeEvent(self, event):
        # Update the space height whenever the widget is resized
        self.update_space_height()
        self.update()

    def draw_placeholder_text(self, painter):
        text_color = Qt.black
        # Define font for placeholder text
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        # Draw placeholder text in the center of the widget
        painter.setPen(text_color)
        painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)
