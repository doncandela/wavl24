"""Module wavl.py in the wavl24 package has some classes and functions to
carry out some elementary wavelet operations.

See: Numerical Recipes Third Ed. by Press et al. (2007), referred to below
as NR3, esp. Sec. 13.10 from which some code in this module was adapted. Note
pywavelets (not used here) is a much more complete and advanced wavelet
package than this module.
"""
THIS_IS = 'wavl.py 2/3/24 D. Candela'

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

""" **********************************************************************
      FUNCTIONS
************************************************************************** """
def getl(nn,nnmin=4):
    """Finds l such that nn = 2**l, errors out if nn<nnmin or nn is
    not a power of 2.
    """
    if nn>=nnmin:
        nn1 = nn
        l = 0
        while not nn1&1:  # while last bit is 0
            nn1 >>= 1
            l += 1
        if nn1==1:
            return l
    raise Exception(f'nn={nn} is not >={nnmin} and a power of 2.')

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


def showdwt(dwt,nnshow=None,asp=0.6):
    """Uses plt.imshow to make a graphical plot of a 1D discrete
    wavelet transform.  For flexibility, does not call plt.figure at
    start or plot.show at end.
    
    In the plot:
        dwt[0..3] will be bottom row, assumed coarsest elements
        dwt[4..7] will be next row up
        dwt[8..15] will be next row up
        dwt[16..31] will be next row up
        ...
    
    Parameters
    ----------
    dwt : array (nn) of float
        1D discrete wavelet transform, errors out if nn not a power
        of 2 and >=4.  dwt could also be some function of the dwt e.g.
        its square.
    nnshow : int or None
        If provided must be a power of two <= nn, will only show
        first nnshow components.
    asp : float
        Desired aspect ratio (height/width)
    
    Side effects
    ------------
    calls plt.imshow
    """
    if nnshow:
        dwt = dwt[:nnshow]
    nn = dwt.size
    l = getl(nn)
    # Width in pixels of the plot will be nn/2, and there will be l-1 rows
    # of height sgv in pixels.  Choose sgv and get array to hold plot.
    nn2 = nn//2       # integer nn/2 = pixels in horizontal direction of plot
    sgv = round(asp*nn2/(l-1))
    ppv = (l-1)*sgv    # pixels in vertical direction of plot
    sg = np.zeros((ppv,nn2))

    jdwt = 0  # index of next element of dwt to be shown
    for row in range(l-1):   # row number counting up from bottom
        # Bottom+1 and top pixels for this row.
        pb = ppv - sgv*row
        pt = pb - sgv
        # Number of columns in this row 4,4,8,16,32...
        cols = max(4,2**(row+1))
        sgh = nn2//cols  # horizontal pixels in each column
        for col in range(cols):
            # Left and right+1 pixels for this column.
            pl = sgh*col
            pr = pl + sgh
            # Fill in pixels for this element of dwt and advance index.
            sg[pt:pb,pl:pr] = dwt[jdwt]
            jdwt += 1    
    plt.imshow(sg)


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
