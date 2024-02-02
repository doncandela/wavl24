"""Module wavl.py in the wavl24 package has some classes and functions to
carry out some elementary wavelet operations.

See: Numerical Recipes Third Ed. by Press et al. (2007), referred to below
as NR3, esp. Sec. 13.10 from which some code in this module was adapted. Note
pywavelets (not used here) is a much more complete and advanced wavelet
package than this module.
"""
THIS_IS = 'wavl.py 2/2/24 D. Candela'

import numpy as np
from numpy import sqrt

""" **********************************************************************
      FUNCTIONS
************************************************************************** """
def wt1(a,wlet,forward=True):
    """One-dimensional discrete wavelet transform.  Adapted from NR3 Sec.
    13.10.2
    
    Parameters
    ----------
    a : array (n) of float
        Input array, which this function replaces by its wavelet transform
        (if forward is true) or inverse transform (otherwise). n must be a
        power of 2.
    wlet: Wavelet
        Wavelet type to use.
    forward : bool
    """
    n = a.size
    if n<4:
        return
    if forward:
        wlet.condition(a,n,True)
        # Start at largest hierarchy, work towards smallest.
        nn = n
        while nn>=4:
            wlet.filt(a,nn,True)
            nn >>= 1
    else:
        # Start at smallest hierarchy, work towards largest.
        nn = 4
        while nn<=n:
            wlet.filt(a,nn,False)
            nn <<= 1
        wlet.condition(a,n,False)


""" **********************************************************************
      BASE CLASSES
************************************************************************** """
class Wavelet:
    """Base class for wavelets. Adapted from NR3 Sec. 13.10.2
    """
    def filt(a,n,forward):
        """Applies wavelet filter (if forward is true) or its transpose
        (otherwise) to a[0..n-1]. Called hierarchically by wt1 and wtn.
        """
        
    def condition(a,n,isign):
        """Pre-conditioning and post-conditioning.
        """

""" **********************************************************************
      CLASSES FOR SPECIFIC WAVELET TYPES
************************************************************************** """
# Daubechies-4 coefficients
D4D = 4*sqrt(2)
D4C0 = (1+sqrt(3))/D4D
D4C1 = (3+sqrt(3))/D4D
D4C2 = (3-sqrt(3))/D4D
D4C3 = (1-sqrt(3))/D4D

class Daub4(Wavelet):
    """Daubechies 4-coefficient wavelet. Adapted from NR3 Sec. 13.10.2
    """
    def filt(a,n,forward):
        """Applies wavelet filter (if forward is true) or its transpose
        (otherwise) to a[0..n-1]. Called hierarchically by wt1 and wtn.
        """
        if n<4:
            return
        ws = np.zeros(n)
        nh = n>>1
        if forward:
            # Apply filter.
            for i,j in enumerate(range(0,n-3,2)):
                ws[i] = D4C0*a[j] + D4C1*a[j+1] + D4C2*a[j+2] + D4C3*a[j+3]
                ws[i+nh] = D4C3*a[j] - D4C2*a[j+1] + D4C1*a[j+2] - D4C0*a[j+3]
        else:
            # Apply transpose filter.
            ws[0] = D4C2*a[nh-1] + D4C1*a[n-1] + D4C0*a[0] + D4C3*a[nh]
            ws[1] = D4C3*a[nh-1] - D4C0*a[n-1] + D4C1*a[0] + D4C2*a[nh]
            for i in range(nh-1):
                j = 2*(i+1)
                ws[j] = (D4C2*a[i] + D4C1*a[i+nh] + D4C0*a[i+1]
                         + D4C3*a[i+nh+1])
                ws[j+1] = (D4C3*a[i] - D4C0*a[i+nh] + D4C1*a[i+1]
                           - D4C2*a[i+nh+1])
        a[:n] = ws[:n]
