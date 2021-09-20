import numpy as np
from sympy.physics.wigner import wigner_6j
import arc


## Common term to alpha0,alpha1 and alpha2
wignerprecal = np.zeros(shape=(7,7,1,21,6,3),dtype='float',order='F')

i=0
for j1 in [1,2,3,4,5,6,7]:
    j=0
    for j2 in [1,2,3,4,5,6,7]:
        k=0
        for j3 in [1]:
            l = 0
            for fsum in range(-10,11):
                m=0
                for f in [0,1,2,3,4,5]:
                    n = 0
                    for Inucl in [3/2,5/2,7/2]:
                        try:
                            wignerprecal[i,j,k,l,m,n] = wigner_6j(j1/2,j2/2,1,fsum,f,Inucl)
                        except ValueError:
                            wignerprecal[i,j,k,l,m,n] = float('NaN')
                    #wignerprecal[i,j,k,1,l,m] = wigner_6j(1,1,1,f,f,fsum)
                    #wignerprecal[i,j,k,1,l,m] = wigner_6j(1,1,2,f,f,fsum)
                        n += 1 
                    m+=1
                l+=1
            k+=1
        j+=1
    i+=1
                    
### Term specific to alpha 1 and alpha 2
wignerprecal12 = np.zeros(shape=(2,6,21),dtype='float',order='F')
i=0
for j3 in [1,2]:
    j = 0
    for f in [0,1,2,3,4,5]:
        k=0
        for fsum in range(-10,11):
            try:
                wignerprecal12[i,j,k]=wigner_6j(1,1,j3,f,f,fsum)
            except ValueError:
                wignerprecal12[i,j,k] = float('NaN')
            k +=1
        j+=1
    i+=1


### Exports    
np.save(r'.\data\precal.npy', wignerprecal)
np.save(r'.\data\precal12.npy', wignerprecal12)

#%%

### To copy paste in atomic system
wignerprecal = np.load("./data/precal.npy",
                         encoding='latin1', allow_pickle=True)

wignerprecal12 = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "data", "precal12.npy"),
                         encoding='latin1', allow_pickle=True)

def wigner6j(j1,j2,j3,m1,m2,m3):
    global wignerprecal
    return wignerprecal[int(2*j1)-1,int(2*j2)-1,j3-1,m1+10,m2,int(np.floor(m3))-2]


def wigner6j12(j1,j2,j3,m1,m2,m3):
    global wignerprecal12
    return wignerprecal12[j3-1,m1,m3+10]