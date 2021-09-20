import sys

from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtWidgets import (
    QApplication,QWidget, QDialog,QAbstractItemView, QTableWidgetItem, QFileDialog, QPushButton, QTreeView, QVBoxLayout, QHBoxLayout
)
from copy import deepcopy


class TestDialog(QDialog):
    def __init__(self, data):

        super(TestDialog, self).__init__()

        self.data = deepcopy(data)

        # Layout
        btOk = QPushButton("OK")
        btCancel = QPushButton("Cancel")
        self.tree = QTreeView()
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btOk)
        hbox.addWidget(btCancel)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.tree)
        self.setLayout(vbox)
        self.setGeometry(300, 300, 600, 400)

        # Button signals
        btCancel.clicked.connect(self.reject)
        btOk.clicked.connect(self.accept)

        # Tree view
        self.tree.setModel(QtGui.QStandardItemModel())
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(True)
        self.tree.setHeaderHidden(False)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectItems)

        self.tree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.tree.model().itemChanged.connect(self.handleItemChanged)

        for x in self.data:
            if not self.data[x]:
                continue
            parent = QtGui.QStandardItem(x)
            parent.setFlags(QtCore.Qt.NoItemFlags)
            for y in self.data[x]:
                value = self.data[x][y]
                child0 = QtGui.QStandardItem(y)
                child0.setFlags(QtCore.Qt.NoItemFlags |
                                QtCore.Qt.ItemIsEnabled)
                child1 = QtGui.QStandardItem(str(value))
                child1.setFlags(QtCore.Qt.ItemIsEnabled |
                                QtCore.Qt.ItemIsEditable |
                                ~ QtCore.Qt.ItemIsSelectable)
                parent.appendRow([child0, child1])
            self.tree.model().appendRow(parent)

        self.tree.expandAll()

    def get_data(self):
        return self.data
    
    def handleItemChanged(self, item):
        parent = self.data[item.parent().text()]
        key = item.parent().child(item.row(), 0).text()
        parent[key] = type(parent[key])(item.text())

class Example(QWidget):

    def __init__(self):

        super(Example, self).__init__()

        btn = QPushButton('Button', self)
        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.show_dialog)

        self.data = {}
        # This example will be hidden (has no parameter-value pair)
        self.data['example0'] = {}
        # A set example with an integer and a string parameters
        self.data['example1'] = {}
        self.data['example1']['int'] = 14
        self.data['example1']['str'] = 'asdf'
        # A set example with a float and other non-conventional type
        self.data['example2'] = {}
        self.data['example2']['float'] = 1.2

    def show_dialog(self):
        dialog = TestDialog(self.data)
        accepted = dialog.exec_()
        if not accepted:
            return
        self.data = deepcopy(dialog.get_data())
        print(self.data)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())