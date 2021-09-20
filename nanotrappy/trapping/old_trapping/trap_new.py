# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:03:07 2020

@author: Adrien
"""

class Trap():
    def __init__(self,nblue,nred,Pblue=1,Pred=1,**kwargs):
        self.nblue = nblue  # 0, 1 or 2)
        self.nred = nred
        self.Pblue = Pblue
        self.Pred = Pred
        labels = []
        directions = []
        wavelengths_trap = []
        if len(kwargs) == nblue+nred:
            for key, value in kwargs.items():
                wavelengths_trap = wavelengths_trap + [value]
                if "blue" in key: 
                    labels = labels + ["blue"]
                elif "red" in key :
                    labels = labels + ["red"]
                else :
                    raise ValueError("The labels should be 'red' or 'blue' ! ")
                if "fwd" in key: 
                    directions = directions + ["fwd"] 
                elif "bwd" in key :
                    directions = directions + ["bwd"]
                else:
                    raise ValueError("The directions should be either 'fwd' or 'bwd' ! ")
            self.directions = directions
            self.labels = labels
            self.wavelengths_trap = wavelengths_trap
        else :
            raise TypeError('The provided number of wavelengths/powers does not match the number of selected beams')

    def set_powers(self,**kwargs):
        if len(kwargs) <= 2:
            self.__dict__.update(kwargs)
        else:
            raise 'The provided number of powers does not match the number of selected beams'

