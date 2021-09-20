from gui.app import App
from PyQt5.QtWidgets import (
    QApplication
)

from qt_material import apply_stylesheet
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    apply_stylesheet(app, theme='dark_cyan.xml')
    win.show()
    sys.exit(app.exec())
