import numpy as np
from nanotrappy.utils.quantumoperators import F0, F0Fm, Fp, Fm, FmFp,F0Fp, F0F0, FmFm, FpFp

def deltascalar(Ep,Em,E0,f):
    return np.eye(2*f+1,2*f+1)*((Em*np.conj(Em) + Ep*np.conj(Ep) + E0*np.conj(E0))/4)

def deltavector(Ep,Em,E0,f):
    f0, fp, fm = F0(f), Fp(f), Fm(f)
    return (1/(2*f))*(2*(Ep*np.conj(Ep)/4-Em*np.conj(Em)/4)*f0 + np.sqrt(2)*((E0*np.conj(Em)/4+Ep*np.conj(E0)/4)*fm + (E0*np.conj(Ep)/4+Em*np.conj(E0)/4)*fp))

def deltatensor(Ep,Em,E0,f):
    f0, fp, fm, f0fm, f0fp, fmfp, f0f0, fmfm, fpfp = F0(f), Fp(f), Fm(f), F0Fm(f), F0Fp(f), FmFp(f), F0F0(f), FmFm(f), FpFp(f)
    # return (3*(E0*np.conj(Em)-Ep*np.conj(E0))/(4*np.sqrt(2)*f*(2*f-1)))*np.dot(f0,fm)+(3*(Em*np.conj(E0)-E0*np.conj(Ep))/(4*np.sqrt(2)*f*(2*f-1)))*np.dot(f0,fp)+((Em*np.conj(Em)+Ep*np.conj(Ep)-2*E0*np.conj(E0))/(8*f*(2*f-1)))*np.dot(fm,fp)+((Em*np.conj(Em)+Ep*np.conj(Ep)-2*E0*np.conj(E0))/(8*f*(2*f-1)))*f0+((-Em*np.conj(Em)-Ep*np.conj(Ep)+2*E0*np.conj(E0))/(4*f*(2*f-1)))*np.dot(f0,f0)+(3*(E0*np.conj(Em)-Ep*np.conj(E0))/(8*np.sqrt(2)*f*(2*f-1)))*fm+(3*(E0*np.conj(Ep)-Em*np.conj(E0))/(8*np.sqrt(2)*f*(2*f-1)))*fp-(3*Ep*np.conj(Em)/(8*f*(2*f-1)))*np.dot(fm,fm)-(3*Em*np.conj(Ep)/(8*f*(2*f-1)))*np.dot(fp,fp)
    return (3*(E0*np.conj(Em)-Ep*np.conj(E0))/(4*np.sqrt(2)*f*(2*f-1)))*f0fm+(3*(Em*np.conj(E0)-E0*np.conj(Ep))/(4*np.sqrt(2)*f*(2*f-1)))*f0fp+((Em*np.conj(Em)+Ep*np.conj(Ep)-2*E0*np.conj(E0))/(8*f*(2*f-1)))*fmfp+((Em*np.conj(Em)+Ep*np.conj(Ep)-2*E0*np.conj(E0))/(8*f*(2*f-1)))*f0+((-Em*np.conj(Em)-Ep*np.conj(Ep)+2*E0*np.conj(E0))/(4*f*(2*f-1)))*f0f0+(3*(E0*np.conj(Em)-Ep*np.conj(E0))/(8*np.sqrt(2)*f*(2*f-1)))*fm+(3*(E0*np.conj(Ep)-Em*np.conj(E0))/(8*np.sqrt(2)*f*(2*f-1)))*fp-(3*Ep*np.conj(Em)/(8*f*(2*f-1)))*fmfm-(3*Em*np.conj(Ep)/(8*f*(2*f-1)))*fpfp

