class material():
    """
        This is the class that implements the different materials that can be used for the structure. A few common materials are already implemented.
        
        Attributes:
            index (float): Refractive index of the material at the frequency of interest
            C3 (dict): C3 coefficients, for a few alkali atoms. Given in SI units.
            
        Example:
            
            >>> print(SiO2().C3)
            {'Cs': 7.8e-49, 'Rb': 4.85e-49}
    """
    def __init__(self, index,C3_SI):

        self.n = index
        self.C3 = C3_SI

    def __str__(self):
        return self.__class__.__name__
    
            
class GaInP(material):
    def __init__(self):
        super().__init__(3.31,{"Rb" : 9.25e-49}) 
        
class SiN(material):
    def __init__(self):
        super().__init__(2,{"Cs" : 13.2e-49}) #see Juan Andres Muniz Silva thesis
        
class SiO2(material):
    def __init__(self):
        super().__init__(1.45246,{"Cs" : 7.80e-49 ,"Rb" : 4.85e-49}) #see Stern Alton NJP 2011 and Solano Grover PRA 2019
    
class air(material):
    def __init__(self):
        super().__init__(1,None)
