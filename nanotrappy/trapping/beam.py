from os import error
import numpy as np
import nanotrappy.utils.physicalunits as pu

class Beams():
    """This is the parent class that implements the beams.

    It is used to define the general tests to distinguish between a Beam and a BeamPair.

    Examples:
        To test if a object object of type Beams is an instance of the subclass BeamPair:

        >>> BP = BeamPair(852e-9,1*mW,852e-9,1*mW)
        >>> BP.isBeam()
        False
        >>> BP.isBeamPair()
        True

    """
    def isBeam(self):
        """Checks if a Beams object is an instance of the Beam subclass.

        Returns:
            bool: True if the object is an instance of the Beam subclass
        """
        if type(self).__name__ == "Beam":
            return True
        else:
            return False
       
    def isBeamPair(self):
        """Checks if a Beams object is an instance of the BeamPair subclass.

        Returns:
            bool: True if the object is an instance of the BeamPair subclass
        """
        if type(self).__name__ == "BeamPair":
            return True
        else:
            return False
        
    def isBeamSum(self):
            """Checks if a Beams object is an instance of the Beam subclass.

            Returns:
                bool: True if the object is an instance of the Beam subclass
            """
            if type(self).__name__ == "BeamSum":
                return True
            else:
                return False
 
class Beam(Beams):
    """This is the subclass that implements a single beam.

    Attributes:
        lmbda (float): wavelength of the beam (m)
        direction (string): direction of the Beam. "f" for a forward beam, "b" for a bacward propagating beam.
        power (float): power of the beam (W)

    Note:
        The direction here is not used in the program, but we found it useful to keep in mind the physical system under study.

    Examples:
        To create an instance of the Beam class:

        >>> BP = Beam(852e-9,"f",1*mW)
        
    """
    def __init__(self,lmbda,direction,power, index = None):
        self.lmbda= lmbda
        self.direction = direction
        self.power = power
        if index == None :
            self.indices = None
        elif isinstance(index, int):
            self.indices =  index
        else : 
            raise ValueError("If an index is specified it has to be integer !")
        
        
    def get_lmbda(self):
        """Getter function for the lmbda attribute

        Returns:
            float: wavelength of the beam
        """
        return self.lmbda

    def get_direction(self):
        """Getter function for the direction attribute

        Returns:
            str: direction of the beam
        """
        return self.direction
    
    def get_power(self):
        """Getter function for the power attribute

        Returns:
            float: power of the beam
        """
        return [self.power]
    
    def get_indices(self):
        return self.indices
    
    def set_power(self, power):
        """Set the power attribute of the beam.

        Args:
            power (float): new power to be set (W)
        """
        self.power = power

    def set_direction(self, direction):
        """Set the direction attribute of the beam.

        Args:
            direction (str): new direction to be set.
        """
        self.direction = direction

    def set_lmbda(self, lmbda):
        """Set the wavelength attribute of the beam.

        Args:
            lmbda (float): new wavelength to be set (m)
        """
        self.lmbda = lmbda


class BeamPair(Beams):
    """This is the subclass that implements a single beam.

    Attributes:
        lmbda1 (float): wavelength of the first beam of the pair (m)
        power1 (float): power of the first beam of the pair (W)
        lmbda2 (float): wavelength of the second beam of the pair (m)
        power2 (float): power of the second beam of the pair (W)

    Caution:
        Here the powers have to be the same for now. It would considerably slow the computation if not.
        We still put two args to mimic the physical system. We hope it will be implemented in the future.

    Caution:
        The wavelengths do not have to be the same, but in the spirit of the current research, we limit the possible difference.
        This boundary wavelength_equality_tolerance can be modified if need in the utils.utils module.

    Raise:
        ValueError: if power1 not equal to power 2
        ValueError: if |lmbda1-lmbda2| > wavelength_equality_tolerance

    Examples:
        To create an instance of the BeamPair class:

        >>> BP = BeamPair(852e-9,"f",1*mW,852e-9,"f",1*mW)
        
    """
    def __init__(self,lmbda1,power1,lmbda2,power2):
        if abs(lmbda1-lmbda2)<pu.wavelength_equality_tolerance:
            # if power1 == power2:
            #     self.beam1 = Beam(lmbda1,"f",power1)
            #     self.beam2 = Beam(lmbda2,"b",power2)
            # else : 
            #     raise ValueError('Cannot set different values of power for a pair of beams.')

            self.beam1 = Beam(lmbda1,"f",power1)
            self.beam2 = Beam(lmbda2,"b",power2)
        else:
            raise ValueError('Cannot set different values of wavelengths for a pair of beams.')
            
    def get_lmbda(self):
        """Get the wavelengths of a BeamPair

        Returns:
            array: an array of the two wavelengths
        """
        return np.array([self.beam1.lmbda, self.beam2.lmbda])
    
    def get_direction(self):
        return np.array([self.beam1.direction, self.beam2.direction])
        
    def get_power(self):
        """Get the powers of a BeamPair

        Returns:
            float: power of the two beams

        Caution:
            As the two powers have to be equal, this functions returns only one float.
        """
        return np.array([self.beam1.power, self.beam2.power])
        # return self.beam1.power
    
    def get_indices(self):
        return np.array([None, None])
    
    def set_lmbda(self, *args):
        """Set the wavelengths of a BeamPair

        If a list is provided, the wavelengths are set idependently using this list. 
        If a float is provided, the two wavelengths are set to be equal.

        Args:
            lmbdas (float): either a list of two values, or one value (m)
        
        Raise:
            ValueError: if |lmbda1-lmbda2| > wavelength_equality_tolerance
            ValueError: if more than 2 wavelengths are provided

        """
        if len(args) == 1 :
            self.beam1.lmbda = args[0]
            self.beam2.lmbda = args[0]
        elif len(args) == 2 :
            if abs(args[0]-args[1])<pu.wavelength_equality_tolerance:
                self.beam1.lmbda = args[0]
                self.beam2.lmbda = args[1]
            else:
                raise ValueError('Cannot set different values of wavelengths for a pair of beams.')
        else :
            raise ValueError("Too many values to unpack")
            
    def set_power(self,*args):
        """Set the powers of a BeamPair

        Only one power is needed as the powers have to be the same.

        Args:
            power (float): new power to be set for the two beams (W)

        """
        if len(args) == 1 :
            self.beam1.power = args[0]
            self.beam2.power = args[0]
        elif len(args) == 2 :
            self.beam1.power = args[0]
            self.beam2.power = args[1]
        else:
            raise ValueError("You must give either 1 or 2 powers.")

class BeamSum(Beams):
    """This is the subclass that implements a sum of beams. It can be useful when you want to tune the relative powers between the different beams

        >>> BS = BeamSum(852e-9,1*mW,852e-9,1*mW)
        
    """
    def __init__(self,lmbdas,powers, indices = None):
        if indices == None :
            self.indices = np.array([None for k in range(len(lmbdas))])
        elif len(indices) == len(lmbdas):
            self.indices =  np.array(indices)
        else : 
            raise ValueError("Please specify as many indices as wavelengths in the BeamSum object or not at all")
        
        self.beams = []
        if len(lmbdas) != len(powers) :
            raise ValueError("Please specify as many wavelengths as powers in the BeamSum object")
        for k in range(len(lmbdas)):
            self.beams += [Beam(lmbdas[k],"f", powers[k])]
            
    def get_lmbda(self):
        """Get the wavelengths of a BeamPair

        Returns:
            array: an array of the two wavelengths
        """
        return np.array([beam.lmbda for beam in self.beams])
    
    def get_direction(self):
        raise ValueError("In a BeamSum all beams are 'forward'")
        
    def get_indices(self):
        return self.indices
        
    def get_power(self):
        """Get the powers of a BeamSum

        Returns:
            list: power of each beam
        """
        return np.array([beam.power for beam in self.beams])
    
    def set_lmbda(self, *args):
        return
            
    def set_power(self,powers):
        """Set the powers of a BeamPair

        Only one power is needed as the powers have to be the same.

        Args:
            power (float): new power to be set for the two beams (W)

        """
        
        if isinstance(powers, float) :
            powers = powers*np.ones(len(self.beams))
            
        if len(powers) != len(self.beams) :
            raise ValueError("Please specify as many new powers as there are beams in the BeamSum object")

        for k in range(len(self.beams)):
            self.beams[k].power = powers[k]