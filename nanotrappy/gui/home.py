from PyQt5.QtWidgets import *
from PyQt5.QtCore import (QPoint, Qt, pyqtSignal)
from NanoTrap.gui.home_ui import Ui_Home
from NanoTrap.gui.app import App
from qt_material import apply_stylesheet

class HomePage(QMainWindow, Ui_Home):

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setStyleSheet(open("./assets/home_stylesheet.qss", "r").read())
        self.pushButton.clicked.connect(self.clicked.emit)
    
    
class MyApp(QMainWindow):
    def __init__(self, parent=None):
        '''
        Constructor
        '''
        super().__init__(parent)
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
    
        self.start_screen = HomePage(self)
        self.second_screen = App(self)

        self.central_widget.addWidget(self.start_screen)
        self.central_widget.addWidget(self.second_screen)
        self.central_widget.setCurrentWidget(self.start_screen)

        self.start_screen.clicked.connect(self.hometoapp)
        self.second_screen.clicked.connect(self.apptohome)

        self.start_screen.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.second_screen.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.xinit = 362
        self.yinit = 133
        
    # def moveEvent(self, e):
    #     print(self.pos())
    #     self.xinit = self.pos().x()
    #     self.yinit=self.pos().y()
    #     super().moveEvent(e)

    def hometoapp(self):
        self.central_widget.setCurrentWidget(self.second_screen)
    
        self.second_screen.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.start_screen.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.move(200, 0)
        self.central_widget.adjustSize()
        self.adjustSize()
    
    def apptohome(self):
        self.central_widget.setCurrentWidget(self.start_screen)

        self.start_screen.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.second_screen.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    
        self.move(self.xinit, self.yinit)
        self.central_widget.adjustSize()
        self.adjustSize()
        
        

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)    # <---
    window = MyApp()
    window.resize(640, 400)
    window.show()
    sys.exit(app.exec_())