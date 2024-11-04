from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import (
    QComboBox,
)


class CheckableComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addItem("Select Models...")  # Placeholder text
        self.model().item(0).setEnabled(False)
        self.setStyleSheet("QComboBox { color: black; }")
        self.model().itemChanged.connect(
            self.handleItemChanged
        )  # Connect signal to slot

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Checked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def returnSelectedItems(self):
        selected_models = []
        for i in range(self.count()):  # Iterate by index
            item = self.model().item(i)
            if item.checkState() == Qt.Checked and item.text() != "Select Models...":
                selected_models.append(item.text())
        return selected_models

    def handleItemChanged(self, item, parent=None):
        # This method will be called whenever an item's check state changes
        if item.checkState() == Qt.Checked:
            print(f"{item.text()} checked")

        else:
            print(f"{item.text()} unchecked")
