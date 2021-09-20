from NanoTrap.utils.physicalunits import *
from NanoTrap.utils.shiftmatrices import deltascalar, deltavector, deltatensor
import os
#from arc import Rubidium87, Caesium
from sympy.physics.wigner import wigner_6j
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
import arc

utils_path = os.path.split(os.path.dirname(__file__))[0] + r"/utils"

wignerprecal = np.load(utils_path + r"/precal.npy", allow_pickle=True)
wignerprecal12 = np.load(utils_path + r"/precal12.npy", allow_pickle=True)

def wigner6j(j1,j2,j3,m1,m2,m3):
    global wignerprecal
    return wignerprecal[int(2*j1)-1,int(2*j2)-1,int(j3-1),int(m1+10),int(m2),int(np.floor(m3))-1]

def wigner6j12(j1,j2,j3,m1,m2,m3):
    global wignerprecal12
    return wignerprecal12[int(j3-1),int(m1),int(m3+10)]


def asfraction(x):
    if x == 0.5:
        return "1/2"
    elif x == 1.5:
        return "3/2"
    elif x == 2.5:
        return "5/2"
    else :
        return "Not implemented"

class atomiclevel():
    """This is the class that implements the atomic levels.

    An atomiclevel object is used as a container for the three quantum numbers n, l and j of the considered level

    Note:
        Some method to define an order between levels is implemented, even though it is not in use right now.

    Attributes:
        n (int): Principal quantum number.
        l (int): Azimuthal (orbital angular momentum) quantum number.
        j (int or half-int) : Total angular momentum quantum number.

    Examples:
        In order to create an atomic level, one can just write:

        >>> level = atomiclevel(6,P,1/2)
        >>> print(level)
        6P1/2

    """
    def __init__(self,n,l,j):
        self.n = n
        self.l = l
        self.j = j


    def __str__(self):
        """ str : Formatted print of an atomic level, using the usual S, P, D... notation. """
        if self.l==0:
            return str(self.n)+"S"+asfraction(self.j)
        elif self.l==1:
            return str(self.n) + "P" + asfraction(self.j)
        elif self.l==2:
            return str(self.n) + "D" + asfraction(self.j)

    def __lt__(self,other):
        if self.n < other.n:
            return True
        elif self.n == other.n:
            if self.j < other.j:
                return True
            elif self.j == other.j:
                if self.l < other.l:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def __eq__(self,other):
        if (self.n == other.n and self.l == other.l and self.j == other.j):
            return True
        else:
            return False

def convert_to_float(frac_str):
    """float: util function to convert string of fraction to float. """
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def string_to_level(str):
    """ atomiclevel: parse an input string into an atomic level object. """
    j = convert_to_float(str[-3::])
    ltemp = str[-4::-3]
    if ltemp =="S":
        l = 0
    elif ltemp =="P":
        l = 1
    elif ltemp =="D":
        l = 2
    elif ltemp =="F":
        l = 3
    elif ltemp =="G":
        l = 4
    else:
        raise "Has not been implemented yet."
    n = int(str[0::-4])
    return atomiclevel(n,l,j)

def check_and_parse(atlevel):
    if isinstance(atlevel,atomiclevel):
        return atlevel
    elif isinstance(atlevel,str):
        return string_to_level(atlevel)
    else:
        raise "Wrong type of arguments."
        
class atomicsystem_dico():
    """This is the class that implements the atomic system under study.
    It calls the ARC Rydberg calculator in the background to collect all the datas available for the chosen atom.

    All the information of the states to which the ground state and the excited state couple to are stored in a array of dictionnaries.
    dicoatom[0] is the dictionnary for the ground state, dicoatom[1] for the excited state
    The coupled states are given by a triplet (n,l,j).

    Note:
        Some method to define an order between levels is implemented, even though it is not in use right now.

    Attributes:
        atom (atom): The atom selected among ARC catalog (ex: Caesium(), Rubidium87()...).
        groundstate (atomiclevel): The atomic level considered as the ground state.
        excitedstate (atomiclevel): The atomic level considered as the excited state.
        Nground (int): the principal quantum number of the ground state (for computationnal use).
        listGround (list[atomiclevel]): list of the atomic levels coupled to the ground state.
        listExcited (list[atomiclevel]): list of the atomic levels coupled to the excited state.
        dicoatom (list[dict]): dictionnary of the physical quantities, to avoid using ARC methods every time.
        dicoatom[0]: physical quantities associated with the levels coupled to the ground state.
        dicoatom[1]: physical quantities associated with the levels coupled to the excited state.

    Note:
        The format of the dictionnaries is the following: To each atomic level is associated a triplet (f,rde,gamma) where f is the transition frequency to this level, rde is the reduced matrix element of the transition and gamma is the transition rate.

    Examples:
        An atomic system is created as follow:

        >>> atomic_system = atomicsystem_dico(Rubidium87(),atomiclevel(5,S,1/2),atomiclevel(5,P,3/2))

        Additionnaly, a parser is provided so that the following is also valid:

        >>> atomic_system = atomicsystem_dico(Rubidium87(),"5S1/2","5P3/2")


    """
    def __init__(self,atom,groundstate,excitedstate):
        self.atom = atom
        self.groundstate = check_and_parse(groundstate)
        self.excitedstate = check_and_parse(excitedstate)

        self.Nground = self.atom.groundStateN

        self.listGround = []
        self.listExcited = []


        self.dicoatom = [{},{}]
    #        self.dicoatom = {self.groundstate:{},self.excitedstate:{}}
        rangeN = 34

        for jnumber in [0.5,1.5]:
            for nnumber in range(self.Nground,self.Nground+rangeN+1):
                self.listGround.append(atomiclevel(nnumber,P,jnumber))
    #                self.dicoatom[self.groundstate][atomiclevel(nnumber,1,jnumber)] = (2*pi*self.atom.getTransitionFrequency(self.groundstate.n,self.groundstate.l,self.groundstate.j,nnumber,1,jnumber),self.atom.getReducedMatrixElementJ(self.groundstate.n,self.groundstate.l,self.groundstate.j,nnumber,1,jnumber),self.atom.getTransitionRate(self.groundstate.n,self.groundstate.l,self.groundstate.j,nnumber,1,jnumber))
                self.dicoatom[0][(nnumber,P,jnumber)] = (self.atom.getTransitionFrequency(self.groundstate.n,self.groundstate.l,self.groundstate.j,nnumber,P,jnumber),self.atom.getReducedMatrixElementJ(self.groundstate.n,self.groundstate.l,self.groundstate.j,nnumber,P,jnumber),self.atom.getTransitionRate(nnumber,P,jnumber,self.groundstate.n,self.groundstate.l,self.groundstate.j))

        ## ground state
        self.listExcited.append(self.groundstate)
    #        self.dicoatom[self.excitedstate][self.groundstate] = self.dicoatom[self.groundstate][self.excitedstate]
        # self.dicoatom[1][(self.groundstate.n,self.groundstate.l,self.groundstate.j)] = self.dicoatom[0][(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j)]
        self.dicoatom[1][(self.groundstate.n,self.groundstate.l,self.groundstate.j)]=(self.atom.getTransitionFrequency(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,self.groundstate.n,self.groundstate.l,self.groundstate.j),self.atom.getReducedMatrixElementJ(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,self.groundstate.n,self.groundstate.l,self.groundstate.j),self.atom.getTransitionRate(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,self.groundstate.n,self.groundstate.l,self.groundstate.j))

       ## S1/2 levels
        jnumber = 1/2
        for nnumber in range(self.Nground+1,self.Nground+rangeN+1):
            self.listExcited.append(atomiclevel(nnumber,S,jnumber))
    #            self.dicoatom[self.excitedstate][atomiclevel(nnumber,0,0.5)]=(2*pi*self.atom.getTransitionFrequency(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,1,jnumber),self.atom.getReducedMatrixElementJ(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,1,jnumber),self.atom.getTransitionRate(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,1,jnumber))
            self.dicoatom[1][(nnumber,S,jnumber)]=(self.atom.getTransitionFrequency(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,S,jnumber),self.atom.getReducedMatrixElementJ(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,S,jnumber),self.atom.getTransitionRate(nnumber,S,jnumber,self.excitedstate.n,self.excitedstate.l,self.excitedstate.j))
        ##D3/2 and D5/2 levels
        for jnumber in [3/2,5/2]:
            for nnumber in range(self.Nground-1,self.Nground+2+rangeN+1):
                self.listExcited.append(atomiclevel(nnumber,D,jnumber))
                #dico[atomiclevel(nnumber,2,1.5)] = (ωtransitionExcitedstate[i],excitedstateRDE[i],decayrateExcitedstate[i]/(2*jnumber +1))
    #                self.dicoatom[self.excitedstate][atomiclevel(nnumber,2,1.5)]=(2*pi*self.atom.getTransitionFrequency(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,1,jnumber),self.atom.getReducedMatrixElementJ(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,1,jnumber),self.atom.getTransitionRate(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,1,jnumber))
                self.dicoatom[1][(nnumber,D,jnumber)]=(self.atom.getTransitionFrequency(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,D,jnumber),self.atom.getReducedMatrixElementJ(self.excitedstate.n,self.excitedstate.l,self.excitedstate.j,nnumber,D,jnumber),self.atom.getTransitionRate(nnumber,D,jnumber,self.excitedstate.n,self.excitedstate.l,self.excitedstate.j))
            #i += 1

    def islower(self,atlevel1,atlevel2):

        """This function checks if atlevel1 is lower in energy than atlevel2. Only used if one of atlevel1 or atlevel2 is either the ground or the excited state

        Args:
            atlevel1 (atomiclevel): atomic level.
            atlevel2 (atomiclevel): atomic level.

        Returns:
            bool: True if successful, False otherwise.

        Example:
            >>> syst.islower(atomiclevel(5,S,1/2),atomiclevel(5,P,3/2))
            True

        """
        if atlevel2 == self.groundstate:
            return False
        elif atlevel2 == self.excitedstate:
            if atlevel1 == self.groundstate:
                return True
            else:
                return False
        

    def alpha0(self,state,f,lmbda):
        """ This function returns the scalar polarizability of the given state and hyperfine state F of the atom, at the wavelength lambda.
        Documentation on how this contribution is calculated is available in the main documentation.

        Args:
            state (atomiclevel): Chosen fine structure atomic level.
            f (int): Hyperfine structure quantum number.
            lmbda (float): Wavelength at which to compute the polarizability.

        Returns:
            float: Scalar polarizability for the given parameters.

        Example:
            >>> syst.alpha0(atomiclevel(5,S,1/2),2,780e-9)/AMU
            -329596.9660348868

        """
        state = check_and_parse(state)
        if f > self.atom.I + state.j or f < abs(self.atom.I - state.j):
            raise ValueError("F should be in the interval [|I-J|,I+J]")
        tot = 0.0
        freq = 0.0
        sign = 1.0
        if state == self.groundstate :
            coupledlevels = self.listGround
            couplings = self.dicoatom[0]
        elif state == self.excitedstate:
            coupledlevels = self.listExcited
            couplings = self.dicoatom[1]
        else:
            return "Not implemented for this level"
        for level in coupledlevels:
            c = couplings[(level.n,level.l,level.j)]
            gamma = c[2]
            omega = 2*np.pi*c[0]
            rde = abs(c[1])

#            rde = abs(self.atom.getReducedMatrixElementJ(state.n,state.l,state.j,level.n,level.l,level.j))
            freq = np.real(1/(sign*omega-(2*np.pi*cc/lmbda)-1j*gamma/2)+1/(sign*omega+(2*np.pi*cc/lmbda)+1j*gamma/2))
            for fsum in np.arange(abs(level.j-self.atom.I),abs(level.j+self.atom.I)+1,1):
                tot += rde*np.conj(rde)*freq*(2*fsum +1)*wigner6j(state.j,level.j,1,fsum,f,self.atom.I)**2
        tot = (1/(3*hbar))*au**2*tot
        if state == self.groundstate :
            tot += 15.8*AMU
        return tot

    def alpha1(self,state,f,lmbda):
        """ This function returns the vector polarizability of the given state and hyperfine state F of the atom, at the wavelength lambda.
        Documentation on how this contribution is calculated is available in the main documentation.

        Args:
            state (atomiclevel): Chosen fine structure atomic level.
            f (int): Hyperfine structure quantum number.
            lmbda (float): Wavelength at which to compute the polarizability.

        Returns:
            float: Vector polarizability for the given parameters.

        """
        if f > self.atom.I + state.j or f < abs(self.atom.I - state.j):
            raise ValueError("F should be in the interval [|I-J|,I+J]")
        tot = 0.0
        freq = 0.0
        sign = 1.0
        if state == self.groundstate :
            coupledlevels = self.listGround
            couplings = self.dicoatom[0]
        elif state == self.excitedstate:
            coupledlevels = self.listExcited
            couplings = self.dicoatom[1]
        else:
            return "Not implemented for this level"
        for level in coupledlevels:
            c = couplings[(level.n,level.l,level.j)]
            gamma = c[2]
            omega = 2*np.pi*c[0]
            rde = abs(c[1])

            freq = np.real(1/(sign*omega-(2*np.pi*cc/lmbda)-1j*gamma/2)-1/(sign*omega+(2*np.pi*cc/lmbda)+1j*gamma/2))
            for fsum in np.arange(abs(level.j-self.atom.I),abs(level.j+self.atom.I)+1,1):
                tot += rde*np.conj(rde)*(-1)**(f+fsum)*np.sqrt(6*f*(2*f+1)/(f+1))*wigner6j12(1,1,1,f,f,fsum)*freq*(2*fsum +1)*wigner6j(state.j,level.j,1,fsum,f,self.atom.I)**2
        tot = (1/hbar)*au**2*tot/2
        return tot

    def alpha2(self,state,f,lmbda):
        """ This function returns the tensor polarizability of the given state and hyperfine state F of the atom, at the wavelength lambda.
        Documentation on how this contribution is calculated is available in the main documentation.

        Args:
            state (atomiclevel): Chosen fine structure atomic level.
            f (int): Hyperfine structure quantum number.
            lmbda (float): Wavelength at which to compute the polarizability.

        Returns:
            float: Tensor polarizability for the given parameters.

        """
        if f > self.atom.I + state.j or f < abs(self.atom.I - state.j):
            raise ValueError("F should be in the interval [|I-J|,I+J]")
        tot = 0.0
        freq = 0.0
        sign = 1.0
        if state == self.groundstate :
            coupledlevels = self.listGround
            couplings = self.dicoatom[0]
        elif state == self.excitedstate:
            coupledlevels = self.listExcited
            couplings = self.dicoatom[1]
        else:
            return "Not implemented for this level"
        for level in coupledlevels:
            c = couplings[(level.n,level.l,level.j)]
            gamma = c[2]
            omega = 2*np.pi*c[0]
            rde = abs(c[1])

            freq = np.real(1/(sign*omega-(2*np.pi*cc/lmbda)-1j*gamma/2)+1/(sign*omega+(2*np.pi*cc/lmbda)+1j*gamma/2))
            for fsum in np.arange(abs(level.j-self.atom.I),abs(level.j+self.atom.I)+1,1):
                tot += rde*np.conj(rde)*(-1)**(f+fsum)*np.sqrt(10*f*(2*f+1)*(2*f-1)/(3*(f+1)*(2*f+3)))*wigner6j12(1,1,2,f,f,fsum)*freq*(2*fsum +1)*wigner6j(state.j,level.j,1,fsum,f,self.atom.I)**2
        tot = (1/(hbar))*au**2*tot
        return tot

    def scalarshift(self,Ep,Em,E0,state,f,lmbda):
        return -1.0*self.alpha0(state,f,lmbda)*deltascalar(Ep,Em,E0,f)

    def scalarshift_contrapropag(self,Ep,Em,E0,state,f,lmbda1, lmbda2):
        alpha0 = (self.alpha0(state,f,lmbda1)+self.alpha0(state,f,lmbda2))/2
        return -1.0*alpha0*deltascalar(Ep,Em,E0,f)

    def vectorshift(self,Ep,Em,E0,state,f,lmbda):
        return -1.0*self.alpha1(state,f,lmbda)*deltavector(Ep,Em,E0,f)

    def vectorshift_contrapropag(self,Ep,Em,E0,state,f,lmbda1,lmbda2):
        alpha1 = (self.alpha1(state,f,lmbda1)+self.alpha1(state,f,lmbda2))/2
        return -1.0*alpha1*deltavector(Ep,Em,E0,f)

    def tensorshift(self,Ep,Em,E0,state,f,lmbda):
        return -1.0*self.alpha2(state,f,lmbda)*deltatensor(Ep,Em,E0,f)

    def tensorshift_contrapropag(self,Ep,Em,E0,state,f,lmbda1,lmbda2):
        alpha2 =(self.alpha2(state,f,lmbda1)+self.alpha2(state,f,lmbda2))/2
        return -1.0*alpha2*deltatensor(Ep,Em,E0,f)

    def totalshift(self,Ep,Em,E0,state,f,lmbda):
        return -1.0*self.alpha0(state,f,lmbda)*deltascalar(Ep,Em,E0,f)-1.0*self.alpha1(state,f,lmbda)*deltavector(Ep,Em,E0,f)-1.0*self.alpha2(state,f,lmbda)*deltatensor(Ep,Em,E0,f)

    def totalshift_contrapropag(self,Ep,Em,E0,state,f,lmbda1, lmbda2):
        alpha0 =(self.alpha0(state,f,lmbda1)+self.alpha0(state,f,lmbda2))/2
        alpha1 =(self.alpha1(state,f,lmbda1)+self.alpha1(state,f,lmbda2))/2
        alpha2 =(self.alpha2(state,f,lmbda1)+self.alpha2(state,f,lmbda2))/2
        return -1.0*alpha0*deltascalar(Ep,Em,E0,f)-1.0*alpha1*deltavector(Ep,Em,E0,f)-1.0*alpha2*deltatensor(Ep,Em,E0,f)

    def potential(self,Ep,Em,E0,state,f,lmbda):
        """ This function computes the trapping potential energy for a given electric field and the given state and hyperfine state F of the atom, at the wavelength lambda.
        Documentation on how this potential is calculated is available in the main documentation.

        Args:
            Ep (float):
            Em (float):
            E0 (float):
            state (atomiclevel): Chosen fine structure atomic level.
            f (int): Hyperfine structure quantum number.
            lmbda (float): Wavelength at which to compute the polarizability.

        Returns:
            float: Trapping potential for the given parameters.

        """
        Htemp = self.totalshift(Ep,Em,E0,state,f,lmbda)/(mK*kB)   ### add vdW later
        vals, vec = LA.eig(Htemp)
        idx = vals.argsort()
        return vals[idx], vec[idx]

    def potential_contrapropag(self,Ep,Em,E0,state,f,lmbda1,lmbda2):
        Htemp = self.totalshift_contrapropag(Ep,Em,E0,state,f,lmbda1,lmbda2)/(mK*kB)   ### add vdW later
        vals, vec = LA.eig(Htemp)
        return vals, vec

    def potential_3beams(self,Ep1,Em1,E01,Ep2,Em2,E02,state,f,lmbda1,lmbda2,lmbda3,vdw = False):
        Htemp = (self.totalshift_contrapropag(Ep1,Em1,E01,state,f,lmbda1,lmbda2)+self.totalshift(Ep2,Em2,E02,state,f,lmbda3))/(mK*kB)
        vals, vec = LA.eig(Htemp)
        return vals, vec

    def potential_4beams(self,Ep1,Em1,E01,Ep2,Em2,E02,state,f,lmbda1,lmbda2,lmbda3,lmbda4,vdw = False):  ### add vdW later
        Htemp = (self.totalshift_contrapropag(Ep1,Em1,E01,state,f,lmbda1,lmbda2)+self.totalshift_contrapropag(Ep2,Em2,E02,state,f,lmbda3,lmbda4))/(mK*kB)   ### add vdW later
        vals, vec = LA.eig(Htemp)
        return vals, vec

    def showLevelMixing(self,Ep,Em,E0,state,f,lmbda):
        _,vec = self.potential(Ep,Em,E0,state,f,lmbda)
        
        zeemanstates = np.arange(-f,f+1,1)
        fig, ax = plt.subplots()
        im = ax.imshow(abs(vec)**2,cmap="viridis")

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(zeemanstates)))
        ax.set_yticks(np.arange(len(zeemanstates)))
        ax.set_xlim(-0.5,len(zeemanstates)-0.5)
        ax.set_ylim(-0.5,len(zeemanstates)-0.5)
        # ... and label them with the respective list entries
        ax.set_xticklabels(zeemanstates)
        ax.set_yticklabels(-zeemanstates)

        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #        rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(zeemanstates)):
            for j in range(len(zeemanstates)):
                text = ax.text(j, i, round((abs(vec)**2)[i, j],2),
                            ha="center", va="center", color="w")

        ax.set_title("Mixing of the Zeeman states")
        fig.colorbar(im)
        fig.tight_layout()
        plt.show()

    def alphaim0(self, state, omega):
        alphacomplex = np.zeros((len(omega),))

        if state == self.groundstate :
            coupledlevels = self.listGround
            couplings = self.dicoatom[0]
        elif state == self.excitedstate:
            coupledlevels = self.listExcited
            couplings = self.dicoatom[1]
        else:
            return "Not implemented for this level"

        rangeN = 34
        pol = arc.calculations_atom_single.DynamicPolarizability(self.atom, state.n, state.l, state.j)
        pol.defineBasis(self.groundstate.n, self.groundstate.n + rangeN)
        alpha_core = pol.getPolarizability(700e-9,units='au')[3]
#        alpha_core = 11
        omega_c = 6e15

        for w in range(len(omega)):
            tot = 0
            for level in coupledlevels:
                c = couplings[(level.n,level.l,level.j)]
#                gamma = c[2]
                omegares = 2*np.pi*c[0]
                rde = abs(c[1])/np.sqrt(2*state.j+1)
#                print(rde)
#                print(omegares)

                tot +=  2/(3*hbar*4*np.pi*ϵ0*a0**3)*(rde*ee*a0)**2*omegares/((omegares)**2+omega[w]**2)
            if state == self.groundstate :
                tot += alpha_core/(1+(omega[w]/omega_c)**2)
            alphacomplex[w]= tot
#        tot = (1/(3*hbar))*au**2*tot

        return alphacomplex


    def get_C3(self,material,state,units="SI"):
        """
        Method to compute C3 coefficient of Casimir-Polder interactions. The calculation assumes an infinite plane
        wall of the given material and with the atom defined by atomicsystem. A database for the refractive of a few
        materials is already implemented but you can add the one you want on the "refractiveindexes" folder.

        Args:
            material (material):
            state (atomiclevel): Atomic level from which we want to compute C3.
            units (str,optional): "SI" or "au" atomic units. Default is SI

        Examples:

            >>>self.get_C3(self.structure.material,self.groundstate)
            >>>5.0093046822224805e-49
        """
        if str(material.__class__.__name__) != "str":
            material = str(material.__class__.__name__)

        xi = np.logspace(1,18,400,base=10) #        lambdalist = [(2*np.pi*cc)/k for k in xi]
        alphaim = self.alphaim0(state,xi)*(a0**3)
        if material == "metal" :
            trap = [(xi[k+1]-xi[k])*(alphaim[k]+alphaim[k+1])/2 for k in range(len(xi)-1)]
        else:
            n = np.load(utils_path + r"/refractiveindexes/" + material + ".npy") #these files contain 3 columns : lambda, Re(n), Im(n) #            Imeps = 2*n[:,1]*n[:,2]
            Reeps = n[:,1]**2-n[:,2]**2

            omegalist = 2*np.pi*cc/(n[:,0]*1e-6)
            trap = np.zeros((len(omegalist),len(xi))) #If the extinction factor is 0, this expression has to be used instead of the standard one
            def integrand_real(k,i):
                return (Reeps[k]-1)*xi[i]/(omegalist[k]**2+xi[i]**2)

            def integrandC3(k):
                return alphaim[k]*((epsilontot[k]-1)/(epsilontot[k]+1))

            for k in range(len(omegalist)-1):
                for i in range(len(xi)):
                    trap[k][i] = (omegalist[k+1]-omegalist[k])*(integrand_real(k,i)+integrand_real(k+1,i))/2

            I = np.sum(trap,axis=0)
            epsilontot = 1 + 2/np.pi * (-I)
            trap = [(xi[k+1]-xi[k])*(integrandC3(k)+integrandC3(k+1))/2 for k in range(len(xi)-1)]

        if units == "SI":
            C3 = hbar/(4*np.pi)*np.sum(trap)
        if units == "au":
            C3 = hbar/((4*np.pi)*np.sum(trap)*(a0**3*EH))
        return C3




# class transitions:
#     def __init__(self,dico):
#         self.transitions_dico = dico

#     def show(self):
#         print("Atomic level => Transition wavelength (2π Hz), Reduced dipole element (au)\n")
#         print("-------------------------------------------------------------------------\n")
#         for atlevel in sorted(self.transitions_dico.keys):
#             if atlevel.l==0:
#                 print(str(atlevel.n),"S",str(asfraction(atlevel.j))," => ", str(self.transitions_dico[atlevel]))
#             elif atlevel.l==1:
#                 print(str(atlevel.n),"S",str(asfraction(atlevel.j))," => ", str(self.transitions_dico[atlevel]))
#             elif atlevel.l==2:
#                 print(str(atlevel.n),"S",str(asfraction(atlevel.j))," => ", str(self.transitions_dico[atlevel]))
