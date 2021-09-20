import nanotrappy.trapping.atomicsystem as Na
from arc import *

import numpy.testing as test

class Testalpha:
    def test_scalar(self):        
        syst = Na.atomicsystem(Caesium(), Na.atomiclevel(6,0,1/2),4)
        print("Testing alpha_scalar")
        test.assert_(syst.alpha_scalar(800e-9) == -3.8803624660287837e-38)
        
    def test_vector(self):        
        syst = Na.atomicsystem(Caesium(), Na.atomiclevel(6,0,1/2),4)
        print("Testing alpha_scalar")
        test.assert_(syst.alpha_vector(800e-9) == -3.8803624660287837e-38)
        

