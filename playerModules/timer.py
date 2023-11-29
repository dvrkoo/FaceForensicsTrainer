from PyQt5.QtCore import QTimer, QObject, pyqtSignal


class VideoTimer(QObject):
    timeout_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.handle_timeout)

    def start(self, interval):
        self.timer.start(interval)

    def stop(self):
        self.timer.stop()

    def handle_timeout(self):
        self.timeout_signal.emit()
