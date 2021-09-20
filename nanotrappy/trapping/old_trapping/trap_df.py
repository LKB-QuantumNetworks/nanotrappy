# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:03:07 2020

@author: Adrien
"""
import pandas as pd

class Trap():
    def __init__(self,nblue,nred,Pblue,Pred,lambdas):
        self.nblue = nblue  # 0, 1 or 2)
        self.nred = nred
        self.Pblue = Pblue
        self.Pred = Pred
        label = []
        direction = []
        wavelengths = pd.Series(list(lambdas.values()))
        for k in range(len(list(lambdas.keys()))):
            if "blue" in list(lambdas.keys())[k]: 
                label = label + ["blue"]
            else :
                label = label + ["red"]
            if "fwd" in list(lambdas.keys())[k]: 
                direction = direction + ["fwd"] 
            else :
                direction = direction + ["bwd"]
        self.lambdas_trap = pd.DataFrame({'lambda': wavelengths, 'label' : label, 'direction': direction})
    def set_powers(self,**kwargs):
        if len(kwargs) <= 2:
            self.__dict__.update(kwargs)
        else:
            raise 'The provided number of powers does not match the number of selected beams'
