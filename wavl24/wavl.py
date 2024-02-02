"""Module wavl.py in the wavl24 package has some classes and functions to
carry out some elementary wavelet operations.

See: Numerical Recipes Third Ed. by Press et al. (2007), referred to below
as NR3, esp. Sec. 13.10 from which some code in this module was adapted. Note
pywavelets (not used here) is a much more complete and advanced wavele
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
# Daubechie-4 coefficients
D4D = 4*sqrt(2)
D4C0 = (1+sqrt(3))/D4D
D4C1 = (3+sqrt(3))/D4D
D4C2 = (3-sqrt(3))/D4D
D4C3 = (1-sqrt(3))/D4D

class Daub4(Wavelet):
    """Daubechie 4-coefficient wavelet.
    """
    def filt(a,n,forward):
        """Applies wavelet filter (if forward is true) or its transpose
        (otherwise) to a[0..n-1]. Called hierarchically by wt1 and wtn.
        """