import numpy as np


def mysqrt(x):
    if x>=0:
        return np.sqrt(x)
    else:
        return 0

def F0(f):
    return np.diag([m for m in range(-f,f+1)])

def Fp(f):
    fp = np.zeros((2*f+1, 2*f+1))
    temp = np.array([np.sqrt(f*(f + 1) - m*(m + 1)) for m in range(-f,f)])
    np.fill_diagonal(fp[1:], temp)
    return fp

def Fm(f):
    fm = np.zeros((2*f+1, 2*f+1))
    temp = np.array([np.sqrt(f*(f + 1) - m*(m - 1)) for m in range(-f+1,f+1)])
    np.fill_diagonal(fm[:,1:], temp)
    return fm

def F0Fp(f):
    return np.array([[(1+m1)*np.sqrt(f*(f+1)-m1*(m1+1))*(1+m1==m2) for m1 in range(-f,f+1,1)] for m2 in range(-f,f+1,1)])

def F0Fm(f):
    return np.array([[(-1+m1)*np.sqrt(f*(f+1)-m1*(m1-1))*(-1+m1==m2) for m1 in range(-f,f+1,1)] for m2 in range(-f,f+1,1)])

def FmFp(f):
    return np.array([[(f*(1 + f) - m1*(1 + m1))*(m1==m2) for m1 in range(-f,f+1,1)] for m2 in range(-f,f+1,1)])

def F0F0(f):
    return np.diag([m**2 for m in range(-f,f+1)])

def FmFm(f):
    return np.array([[mysqrt(f *(1 + f) - (-2 + m1)* (-1 + m1)) *mysqrt(f *(1 + f) - (-1 + m1) *m1)*(m1-2==m2) for m1 in range(-f,f+1,1)] for m2 in range(-f,f+1,1)])

def FpFp(f):
    return np.array([[mysqrt(f *(1 + f) - (1 + m1)* m1) *mysqrt(f *(1 + f) - (1 + m1) *(m1+2))*(m1+2==m2) for m1 in range(-f,f+1,1)] for m2 in range(-f,f+1,1)])

