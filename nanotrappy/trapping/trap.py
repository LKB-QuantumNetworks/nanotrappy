from nanotrappy.utils.physicalunits import *
from os import error
        
class Trap_beams():
    """A class to represent a trap, wich is a bundle of beams and a propagation axis.

    Attributes:
        beams: An array of Beams instances.
        propagation_axis (str): A string that represents the propagation axis of the trap.
    
    Raise:
        ValueError: if no beam is provided 
        ValueError: if 3 Beams are provided, please put two of them in a Pair. 

    TODO:
        Add a way to put as many beams as wanted in a trap.

    Examples:
        To create a trap, first create the instances of Beams:

        >>> BP = BeamPair(935e-9,"f",1*mW,935e-9,"f",1*mW)
        >>> B = Beam(685e-9,"f",1*mW)
        >>> trap = Trap(BP,B,propagation_axis="X")
        
    """
    def __init__(self,*args,propagation_axis="X"):
        self.beams = np.array([])
        for arg in args :
            if arg is not None:  ## Needed for the GUI
                self.beams = np.append(self.beams,arg)
        if len(self.beams) == 0 :
            raise ValueError("You have to input at least one beam in the trap !")
        # if len(self.beams) >2 :
        #     raise ValueError("If you want more than 2 beams, you have to use counterpropgating beams (BeamPair class) !")
        self.propagation_axis=propagation_axis
            
    @property
    def lmbdas(self):
        """Get the wavelengths of each beam in the trap

        Returns:
            array: wavelengths of each beam in the trap (m)
        """
        lmbdas = np.array([])
        for beam in self.beams :
            lmbdas = np.append(lmbdas,beam.get_lmbda())
        return lmbdas
    
    @property
    def powers(self):
        """Get the powers of each beam in the trap

        Returns:
            array: powers of each beam in the trap (W)
        """
        powers = np.array([])
        for beam in self.beams :
            powers = np.append(powers,beam.get_power())
        return powers
    
    @property
    def directions(self):
        """Get the directions of each beam in the trap

        Returns:
            array: directions of each beam in the trap 
        """
        directions = np.array([])
        for beam in self.beams :
            directions = np.append(directions,beam.get_direction())
        return directions
    
    @property
    def indices(self):
        "Index of the file we want to use in the folder"
        indices = np.array([], dtype = int)
        for beam in self.beams :
            indices = np.append(indices,beam.get_indices())
        return indices
        
    def set_powers(self, powerlist):
        """Set the powers of each beam in the trap

        Args:
            powerlist (list): list of one power per Beam (W)

        Raises:
            ValueError: too many powers provided

        Caution:
            Only provide one power for a BeamPair.

        """
        if len(powerlist) == len(self.beams): 
            for beam in self.beams :
                beam.set_power(powerlist.pop(0))
        else :
            raise ValueError("You must give as many powers as there are beams")
