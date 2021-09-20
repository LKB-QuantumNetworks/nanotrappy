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
        if len(kwargs) == nblue+nred:
            #self.__dict__.update(kwargs)
            self.wavelengths_dict = kwargs
        else:
            raise 'The provided number of wavelengths/powers does not match the number of selected beams'

    def set_powers(self,**kwargs):
        if len(kwargs) <= 2:
            self.__dict__.update(kwargs)
        else:
            raise 'The provided number of powers does not match the number of selected beams'

t = Trap(nblue = 2 , nred = 2, lbd1 = 780, lbd2=780, lbd3 = 835,lbd4=835)