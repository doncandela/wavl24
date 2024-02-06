"""Module wavl.py in the wavl24 package has some classes and functions to
carry out some elementary wavelet operations.

See: Numerical Recipes Third Ed. by Press et al. (2007), referred to below
as NR3, esp. Sec. 13.10 from which some code in this module was adapted. Note
pywavelets (not used here) is a much more complete and advanced wavelet
package than this module.
"""
THIS_IS = 'wavl.py 2/6/24 D. Candela'

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

def wt1(a,wavelet,forward=True):
    """One-dimensional discrete wavelet transform.  Adapted from NR3 Sec.
    13.10.2
    
    Parameters
    ----------
    a : array (nn) of float
        Input array, which this function replaces by its wavelet transform
        (if forward is true) or inverse transform (otherwise). nn must be a
        power of 2.
    wavelet: Wavelet
        Wavelet filter to use.
    forward : bool
    """
    nn = a.size
    getl(nn,nnmin=1)
    if nn<4:
        return
    if forward:
        wavelet.condition(a,nn,True)
        # Start at largest hierarchy, work towards smallest.
        n = nn
        while n>=4:
            wavelet.filt(a,n,True)
            n >>= 1
    else:
        # Start at smallest hierarchy, work towards largest.
        n = 4
        while n<=nn:
            wavelet.filt(a,n,False)
            n <<= 1
        wavelet.condition(a,nn,False)

def showdwt(dwt,nnshow=None,asp=0.6):
    """Uses plt.imshow to make a graphical plot of a 1D discrete
    wavelet transform.  For flexibility, does not call plt.figure at
    start or plot.show at end.
    
    In the plot:
        dwt[0..3] will be bottom row, assumed coarsest elements
        dwt[4..7] will be next row up
        dwt[8..15] will be next row up
        dwt[16..31] will be next row up
        etc.
    
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
    nn2 = max(4,nn//2)       # pixels in horizontal direction of plot
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
       CLASSES FOR WAVELET TYPES
************************************************************************** """
class Wavelet:
    """Base class for wavelets. Adapted from NR3 Sec. 13.10.2
    """
    def filt(self,a,n,forward):
        """Applies wavelet filter (if forward is true) or its transpose
        (otherwise) to a[0..n-1]. Called hierarchically by wt1 and wtn.
        """       
    def condition(self,a,nn,forward):
        """Pre-conditioning and post-conditioning.
        """

def getfilter(tp):
    """Helper function: Gets specified wavelet filter.

    Parameters
    ----------
    tp : None or int
        If None, use Daubechies-4 filter from class Daub4. Otherwise use
        Daubechies-tp filter from class Daubs, with tp=4,6...
        
    Returns
    -------
    wavelet : Wavelet
    """
    if tp:
        return Daubs(tp)
    return Daub4()


def daubccs(tp):
    """Helper for Daubechies wavelet classes: Returns the tp = 2*p filter
    coefficients for an order=p filter.
    
    This version only works for tp=4 or 6 (the two cases for which there are
    analytical expressions for the coefficients).
    
    Parameters
    ----------
    tp : 4 or 6
        Number of coefficients returned.
    
    Returns
    -------
    cc : array (tp) of float
        Daubechies filter coefficients.
    """
    if tp==4:
        cc = np.zeros(4)
        d4d = 4*sqrt(2)
        cc[0] = (1+sqrt(3))/d4d
        cc[1] = (3+sqrt(3))/d4d
        cc[2] = (3-sqrt(3))/d4d
        cc[3] = (1-sqrt(3))/d4d
        return cc
    elif tp==6:
        cc = np.zeros(6)
        return cc
    else:
        raise Exception(f'Requested Daubechies size {tp} must be 4 or 6.')

class Daub4(Wavelet):
    """Daubechies 4-coefficient wavelet. Adapted from NR3 Sec. 13.10.2
    """
    def __init__(self):
        self.ccs = daubccs(4)   # get and save D4 filter coefficients
        
    def filt(self,a,n,forward):
        """Applies wavelet filter (if forward is true) or its transpose
        (otherwise) to a[0..n-1]. Called hierarchically by wt1 and wtn.
        """
        cc0,cc1,cc2,cc3 = self.ccs
        if n<4:
            return
        ws = np.zeros(n)
        nh = n>>1
        if forward:
            # Apply filter.
            for i,j in enumerate(range(0,n-3,2)):
                ws[i] = cc0*a[j] + cc1*a[j+1] + cc2*a[j+2] + cc3*a[j+3]
                ws[i+nh] = cc3*a[j] - cc2*a[j+1] + cc1*a[j+2] - cc0*a[j+3]
            ws[i] = cc0*a[n-2] + cc1*a[n-1] + cc2*a[0] + cc3*a[1]
            ws[i+nh] = cc3*a[n-2] - cc2*a[n-1] + cc1*a[0] - cc0*a[1]
        else:
            # Apply transpose filter.
            ws[0] = cc2*a[nh-1] + cc1*a[n-1] + cc0*a[0] + cc3*a[nh]
            ws[1] = cc3*a[nh-1] - cc0*a[n-1] + cc1*a[0] + cc2*a[nh]
            for i in range(nh-1):
                j = 2*(i+1)
                ws[j] = cc2*a[i] + cc1*a[i+nh] + cc0*a[i+1] + cc3*a[i+nh+1]
                ws[j+1] = cc3*a[i] - cc0*a[i+nh] + cc1*a[i+1] - cc2*a[i+nh+1]
        a[:n] = ws[:n]


class Daubs(Wavelet):
    """Daubechies tp-coefficient wavelet for tp=4,6... Adapted from NR3
    Sec. 13.10.2
    
    Parameters
    ----------
    tp : int
        Number of coefficients 4,6... = 2*order of filter.  Allowed values
        are those allowed by daubscc.
    """
    def __init__(self,tp):
        raise NotImplementedError


""" ******************** end of module wavl.py *************************** """