import sys
from PyQt5.QtCore import QFile, QThread, pyqtSignal

from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDialog, QMainWindow, QMessageBox, QPushButton, QSpinBox, QTableWidgetItem, QFileDialog
)
from PyQt5.uic import loadUi
from functools import partial
from qt_material import apply_stylesheet

from NanoTrap.gui.Nanotrap_ui import Ui_MainWindow
import NanoTrap.trapping.atomicsystem as Na
import NanoTrap.trapping.beam as Nb
import NanoTrap.trapping.trap as Nt
import NanoTrap.trapping.simulation as Ns
import NanoTrap.utils.materials as Nm
from NanoTrap.utils.utils import set_axis_from_plane, set_axis_from_axis, check_mf
from NanoTrap.utils import vdw
import NanoTrap.utils.viz as Nv
from NanoTrap.utils.physicalunits import *
from NanoTrap.gui.dictdialog import TestDialog
from arc import *

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import json
from datetime import datetime
import importlib, inspect
from copy import deepcopy
import time

# Setting up common ressources.
def parse_atom(atom):
    if atom=="Rb":
        return Rubidium87()
    if atom=="Cs":
        return Caesium()

def parse_material(mat):
    material_class = getattr(Nm, mat)
    material = material_class()
    return material

def parse_hyperfine_state(f):
    return int(f)

test_params_nanofibre = {}
        # A set example with an integer and a string parameters
test_params_nanofibre['Nanofibre'] = {}
test_params_nanofibre['Nanofibre']['Radius'] = 250
test_params_nanofibre['Nanofibre']['Axis'] = 'X'

progressSettings = """QProgressBar#progressBar {
  border-radius: 0;
  background-color: {{secondaryLightColor}};
  text-align: center;
  color: white
}"""
### Main app
class App(QMainWindow, Ui_MainWindow):
    
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        apply_stylesheet(self, theme='dark_cyan.xml')
        self.initialize_ui()
        self.connectSignalsSlots()

    def initialize_ui(self):
        self.data_path='C:/Users/Berroir/Documents/Thèse/Calculs/NanoTrap/trapping/testfolder'
        self.dataLineEdit.setText(self.data_path)
        self.Button2D.setChecked(True)
        self.progressBar.setValue(0)
        self.idx = 0
        
        self.plotPotentialButton.setEnabled(False)
        with open(os.getcwd()+"/assets/app_stylesheet.qss") as file:
            self.setStyleSheet(self.styleSheet() + file.read())
        
        ## Initialize material choice based on existant classes
        self.materialBox.clear()
        for name, cls in inspect.getmembers(importlib.import_module("NanoTrap.utils.materials"), inspect.isclass):
            if name != "material":
                self.materialBox.addItem(name)

    def connectSignalsSlots(self):
        #self.action_pdf.triggered.connect(self.close)
        self.loadDataButton.clicked.connect(self.ONloadDataButton)
        self.dataLineEdit.editingFinished.connect(self.ONdataLineEdit)
    
        self.Button1D.stateChanged.connect(self.ONButton1D)
        self.Button2D.stateChanged.connect(self.ONButton2D)

        self.fillConfigButton.clicked.connect(self.ONFill)
        self.plotBDButton.clicked.connect(self.clicked.emit)
        
        self.power1pair1.valueChanged.connect(self.ONchangePowers)
        self.power2pair1.valueChanged.connect(self.ONchangePowers)
        self.power1pair2.valueChanged.connect(self.ONchangePowers)
        self.power2pair2.valueChanged.connect(self.ONchangePowers)
        
        self.simulateButton.clicked.connect(self.ONSimulate)
        self.plotPotentialButton.clicked.connect(self.ONPlot)
        # self.action_Exit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.about)

    ### Enable/Disable changes oduring simulation
    def enableAll(self):
        self.speciesBox.setEnabled(True)
        self.groundNBox.setEnabled(True)
        self.groundLBox.setEnabled(True)
        self.groundJBox.setEnabled(True)
        self.hyperfineBox.setEnabled(True)
        self.materialBox.setEnabled(True)
        self.dataLineEdit.setEnabled(True)
        self.lambda1pair1.setEnabled(True)
        self.lambda1pair2.setEnabled(True)
        self.lambda2pair1.setEnabled(True)
        self.lambda2pair2.setEnabled(True)
        self.power1pair1.setEnabled(True)
        self.power2pair1.setEnabled(True)
        self.power1pair2.setEnabled(True)
        self.power2pair2.setEnabled(True)
        self.Button2D.setEnabled(True)
        self.Button1D.setEnabled(True)
        self.orthCoord2DBox.setEnabled(True)
        self.planeBox.setEnabled(True)
        self.coord1Box.setEnabled(True)
        self.coord2Box.setEnabled(True)
        self.plotPotentialButton.setEnabled(True)
        self.simulateButton.setEnabled(True)

    def disableAll(self):
        self.speciesBox.setEnabled(False)
        self.groundNBox.setEnabled(False)
        self.groundLBox.setEnabled(False)
        self.groundJBox.setEnabled(False)
        self.hyperfineBox.setEnabled(False)
        self.materialBox.setEnabled(False)
        self.dataLineEdit.setEnabled(False)
        self.lambda1pair1.setEnabled(False)
        self.lambda1pair2.setEnabled(False)
        self.lambda2pair1.setEnabled(False)
        self.lambda2pair2.setEnabled(False)
        self.power1pair1.setEnabled(False)
        self.power2pair1.setEnabled(False)
        self.power1pair2.setEnabled(False)
        self.power2pair2.setEnabled(False)
        self.Button2D.setEnabled(False)
        self.Button1D.setEnabled(False)
        self.orthCoord2DBox.setEnabled(False)
        self.planeBox.setEnabled(False)
        self.coord1Box.setEnabled(False)
        self.coord2Box.setEnabled(False)
        self.plotPotentialButton.setEnabled(False)
        self.simulateButton.setEnabled(False)

    ###   Get parameteres from screen
    def get_current_atom(self):
        return str(self.speciesBox.currentText())

    def get_current_state(self):
        return str(self.groundNBox.currentText()+self.groundLBox.currentText()+self.groundJBox.currentText())

    def get_hyperfine_level(self):
        return str(self.hyperfineBox.currentText())

    def get_current_material(self):
        return str(self.materialBox.currentText())

    def get_params_simul(self):
        lambda1pair1 = str(self.lambda1pair1.value())+' nm'
        power1pair1 = str(self.power1pair1.value())+' mW'
        lambda2pair1 = str(self.lambda2pair1.value())+' nm'
        power2pair1 = str(self.power2pair1.value())+' mW'
        
        lambda1pair2 = str(self.lambda1pair2.value())+' nm'
        power1pair2 = str(self.power1pair2.value())+' mW'
        lambda2pair2 =str(self.lambda2pair2.value())+' nm'
        power2pair2 = str(self.power2pair2.value())+' mW'
        
        self.params = {
            "Time of simulation": None,
            "Atomic system": {
                "species":self.get_current_atom(),
                "groundstate": self.get_current_state(),
                "hyperfine level":self.get_hyperfine_level()
            },
            "Material": self.get_current_material(),
            'Trap wavelengths':{
                'lambda 1 pair 1':lambda1pair1,
                'lambda 2 pair 1':lambda2pair1,
                'lambda 1 pair 2':lambda1pair2,
                'lambda 2 pair 2':lambda2pair2
                },
            'Trap powers':{
                'power 1 pair 1':power1pair1,
                'power 2 pair 1':power2pair1,
                'power 1 pair 2':power1pair2,
                'power 2 pair 2':power2pair2
                },
            "Considered state": self.get_current_state(),
            "Geometry":{
                "2D": str(self.Button2D.isChecked()),
                "2D plane":str(self.planeBox.currentText()) if self.Button2D.isChecked() else str(None),
                "2D orthogonal coord": str(self.orthCoord2DBox.value()) if self.Button2D.isChecked() else str(None),
                "1D":str(self.Button1D.isChecked()),
                "1D coord1":str(self.coord1Box.value()) if self.Button1D.isChecked() else str(None),
                "1D coord2":str(self.coord2Box.value()) if self.Button1D.isChecked() else str(None)
            },
            "Data_folder":self.data_path
        }
        
    def save_params_to_json(self):
        now = datetime.now()
        current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    
        self.params['Time of simulation'] =  current_time

        with open('data.json', 'w') as fp:
            json.dump(self.params, fp)

    def set_trap_components(self):
        if self.lambda1pair1.value() and self.power1pair1.value():
            if self.lambda2pair1.value() and self.power2pair1.value():
                self.trap_component_1 = Nb.BeamPair(self.lambda1pair1.value()*1e-9,self.power1pair1.value()*1e-3,self.lambda2pair1.value()*1e-9, self.power2pair1.value()*1e-3)
            else:
                self.trap_component_1 = Nb.Beam(self.lambda1pair1.value()*1e-9,"f",self.power1pair1.value()*1e-3)
        else:
            if self.lambda2pair1.value() and self.power2pair1.value():
                self.trap_component_1 = Nb.Beam(self.lambda1pair1.value()*1e-9,"f",self.power1pair1.value()*1e-3)
            else:
                self.trap_component_1 = None

        if self.lambda1pair2.value() and self.power1pair2.value():
            if self.lambda2pair2.value() and self.power2pair2.value():
                self.trap_component_2 = Nb.BeamPair(self.lambda1pair2.value()*1e-9,self.power1pair2.value()*1e-3,self.lambda2pair2.value()*1e-9, self.power2pair2.value()*1e-3)
            else:
                self.trap_component_2 = Nb.Beam(self.lambda1pair2.value()*1e-9,"f",self.power1pair2.value()*1e-3)
        else:
            if self.lambda2pair2.value() and self.power2pair2.value():
                self.trap_component_2 = Nb.Beam(self.lambda1pair2.value()*1e-9,"f",self.power1pair2.value()*1e-3)
            else:
                self.trap_component_2 = None

    def create_simul(self):
        self.atomicsystem = Na.atomicsystem_dico(parse_atom(self.get_current_atom()), self.get_current_state(),f = parse_hyperfine_state(self.get_hyperfine_level()))
        self.material = getattr(Nm,self.materialBox.currentText())()
        
        self.set_trap_components()
        self.surface = vdw.NoSurface()
        self.trap = Nt.Trap_beams(self.trap_component_1,self.trap_component_2)
        self.simul = Ns.Simulation(self.atomicsystem,self.material,self.trap,self.surface,self.data_path)

    ### Button functions

    def ONloadDataButton(self):
        DirName = QFileDialog.getExistingDirectory(self,'Open a folder',self.data_path)

        if (not len(os.listdir(DirName))==0) and (os.path.isdir(str(DirName))):
            self.dataLineEdit.setText(DirName)
            self.data_path = str(DirName)
        print ('Chosen directory:', self.data_path)
        
    def ONdataLineEdit(self):
        DirName = self.dataLineEdit.text()
        if os.path.isdir(str(DirName)):
            self.data_path = str(DirName)
            print ('Chosen directory:', self.data_path)
        else:
            print ('!!!directory not found!!!')
            self.dataLineEdit.setText(self.data_path)

    def ONButton1D(self):
        self.Button2D.setChecked(not self.Button1D.isChecked())
    
    def ONButton2D(self):
        self.Button1D.setChecked(not self.Button2D.isChecked())

    def signal_accept(self,msg):
        self.progressBar.setValue((100*self.idx +int(msg))/2)
        if int(msg)==99:
            self.idx += 1

    def quitThread(self):
        print("Simulation done in  %.4s s." % (time.time()-self.t0))
        self.thread.quit()
        self.enableAll()

        print("Saving the parameters ...")
        self.simul.save()
        
    def ONSimulate(self):
        self.progressBar.setValue(0)
        self.idx = 0

        print("Getting the parameters ...")
        self.get_params_simul()
        # print("Saving the parameters ...")
        # self.save_params_to_json()
        
        print("Creating the simulation ...")
        self.create_simul()

        f = parse_hyperfine_state(self.get_hyperfine_level())

        # 1 - create Worker and Thread inside the Form
        self.thread = QThread()  # no parent!
        # 2 - Connect Worker`s Signals to Form method slots to post data.
        self.simul._signal.connect(self.signal_accept)
        # 3 - Move the Worker object to the Thread object
        self.simul.moveToThread(self.thread)
        # 4 - Connect Worker Signals to the Thread slots
        self.simul.finished.connect(self.quitThread)
        # 5 - Connect Thread started signal to Worker operational slot method
        if self.Button2D.isChecked():
            self.thread.started.connect(partial(self.simul.compute_potential2,str(self.planeBox.currentText()),self.orthCoord2DBox.value()))
        elif self.Button1D.isChecked():
            self.thread.started.connect(partial(self.simul.compute_potential_1D,str(self.axisBox.currentText()),self.coord1Box.value(),self.coord2Box.value()))
        

        print("Running the simulation ...")
        self.disableAll()
        self.t0 = time.time()
        # 6 - Start the thread
        self.thread.start()

    ### Reset plot if data is changed after pressing GO
    def plot_trap(self,mf = 0):
        self.mf = mf
        plane = str(self.planeBox.currentText())
        coord1, coord2 = set_axis_from_plane(plane, self.simul)
        mf_index = int(self.mf + self.simul.atomicsystem.f)
        trap = np.real(self.simul.total_potential())[:,:,mf_index]
        trap_noCP = np.real(self.simul.total_potential_noCP[:,:,mf_index])
        self.fig, self.ax = plt.subplots()

        self.a = plt.pcolormesh(coord1,coord2,np.transpose(trap),shading = "gouraud",norm = colors.TwoSlopeNorm(vmin=min(np.min(trap_noCP)/2,-0.001), vcenter=0, vmax=max(np.max(trap_noCP)*2,0.001)), cmap = "seismic_r")
        cbar = plt.colorbar(self.a)
        plt.xlabel("%s (nm)" %(plane[0]))
        plt.ylabel("%s (nm)" %(plane[1]))

        plt.title("2D plot of trapping potential in the %s plane" %(plane))

        self.ax.margins(x=0)
        plt.show()

    def plot_trap_1D(self,axis,mf = 0, add_CP = False,Pranges=[]):
        _, self.mf = check_mf(self.simul.atomicsystem.f,mf)
        
        x = set_axis_from_axis(axis, self.simul)
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.27)
        jet = cm = plt.get_cmap('Greys')
        cNorm  = colors.Normalize(vmin=-1, vmax=len(mf))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        self.a = []
        
        mf_index = self.mf + [self.simul.atomicsystem.f]
    
        trap = np.real(self.simul.total_potential())
        trap_noCP = np.real(self.simul.total_potential_noCP)
        
        for k in range(len(mf_index)):
            colorVal = scalarMap.to_rgba(k)
            self.a = self.a + plt.plot(x,trap[:,mf_index[k]], color = colorVal, label = "m$_f$ = %s" %(mf[k]), linewidth = 2 + 3/len(self.simul.mf_all))

        if len(mf) == 1:
            self.b, = plt.plot(x,np.real(self.simul.trap.get_powers()[0]*np.real(self.simul.potentials[0,:,mf_index[0]])), color = 'blue', linewidth = 2)
            self.r, = plt.plot(x,np.real(self.simul.trap.get_powers()[0]*np.real(self.simul.potentials[1,:,mf_index[0]])), color = 'red', linewidth = 2)

        plt.axhline(y=0, color='black', linestyle='--')
        # plt.ylim(-3,3)
        plt.legend()
        plt.xlabel(axis+" (nm)")
        plt.ylabel("E (mK)")
        #plt.ylim(-5e-14,5e-14)
        plt.title("1D plot of trapping potential along %s " %(axis))
        plt.show()
    
    def ONchangePowers(self):
        if hasattr(self,"fig"):
            powers = []
            #mf_index = int(self.mf + self.simul.atomicsystem.f)
            mf_index = self.mf + self.simul.atomicsystem.f
            if self.lambda1pair1.value()!=0:
                powers.append(self.power1pair1.value()*mW)
            if self.lambda2pair1.value()!=0:
                if self.lambda1pair1.value()!=self.lambda2pair1.value():
                    powers.append(self.power2pair1.value()*mW)
            if self.lambda1pair2.value()!=0:
                powers.append(self.power1pair2.value()*mW)
            if self.lambda2pair2.value()!=0:
                if self.lambda1pair2.value()!=self.lambda2pair2.value():
                    powers.append(self.power2pair2.value()*mW)
            self.simul.trap.set_powers(powers)

            if self.Button2D.isChecked():
                trap_2D = self.simul.total_potential()[:,:,mf_index] ## 0 is mf, to be changed later

                self.a.set_array(np.transpose(np.real(self.simul.total_potential_noCP[:,:,mf_index])).ravel())
                self.a.autoscale()
                self.a.set_array(np.transpose(np.real(trap_2D)).ravel())

                self.fig.canvas.draw_idle()

            elif self.Button1D.isChecked():
                trap_1D = self.simul.total_potential()
                for k in range(len(self.mf)):
                    trap_k = trap_1D[:,mf_index[k]]
                    self.a[k].set_ydata(trap_k)

                if len(self.mf) == 1:
                    self.b.set_ydata(np.real(self.simul.trap.beams[0].get_power()*np.real(self.simul.potentials[0])))
                    self.r.set_ydata(np.real(self.simul.trap.beams[1].get_power()*np.real(self.simul.potentials[1])))
                self.fig.canvas.draw_idle()

    def ONPlot(self):
        if self.Button2D.isChecked():
            self.plot_trap(mf=0)
        elif self.Button1D.isChecked():
            self.plot_trap_1D(axis=str(self.axisBox.currentText()),mf=self.simul.mf_all)
    
    #### Other stuff 
    def ONFill(self):
        dialog = TestDialog(test_params_nanofibre)
        apply_stylesheet(dialog, theme='light_cyan.xml')
        accepted = dialog.exec_()
        if not accepted:
            return
        self.results = deepcopy(dialog.get_data())
        print(self.results)

    def about(self):
        QMessageBox.about(
            self,
            "NanoTrap Simulator",
            "<p>A simple simulator for trapping around nanostructure  </p>"
            "<p>This simulator has been built by:</p>"
            "<p>- Jeremy Berroir</p>"
            "<p>- Adrien Bouscal</p>"
            "<p></p>"
            "<p>&copy; 2020-2021</p>"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    #apply_stylesheet(app, theme='dark_cyan.xml')
    win.show()
    sys.exit(app.exec())

