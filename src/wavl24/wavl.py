"""Module wavl.py in the wavl24 package has some classes and functions to
carry out some elementary wavelet operations.

See: Numerical Recipes Third Ed. by Press et al. (2007), referred to below
as NR3, esp. Sec. 13.10 from which some code in this module was adapted. Note
pywavelets (not used here) is a much more complete and advanced wavelet
package than this module.
"""
THIS_IS = 'wavl.py 2/7/24 D. Candela'

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

def showdwt(dwt,nnshow=None,asp=0.6,dots=()):
    """Uses plt.imshow to make a graphical plot of a 1D discrete wavelet
    transform, with a colored square for each basis vector. For flexibility, 
    does not call plt.figure at start or plot.show at end.
    
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
    dots: int,int
        Will draw a red dot in center of rectangles representing these basis
        vectors.
    
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
    # Loop over rows and columns to color a rectangle for each basis vector.
    jdwt = 0  # index of next element of dwt to be shown
    dotxs = []
    dotys = []  # where on image to plot red dots
    for row in range(l-1):   # row number counting up from bottom
        # Bottom+1 and top pixels for this row.
        pb = ppv - sgv*row
        pt = pb - sgv
        # Number of columns in this row 4,4,8,16,32...
        cols = max(4,2**(row+1))
        sgh = nn2//cols  # horizontal pixels in each column
        for col in range(cols):
            # Rotate colums - this gives elements in this plot closer to
            # visual center of corresponding wavelet, not sure why.
            col2 = (col+1)%cols
            # Left and right+1 pixels for this column.
            pl = sgh*col2
            pr = pl + sgh
            # Fill in pixels for this element of dwt.
            sg[pt:pb,pl:pr] = dwt[jdwt]
            # If requested to draw dot on this element, find x and y of dot.
            if jdwt in dots:
                dotxs.append((pl+pr)/2)
                dotys.append((pb+pt)/2)
            jdwt += 1  # advance element index.
    # Put red dots at centers of image elements for specified basis vecs.
    if dotxs:
        plt.scatter(dotxs,dotys,color='red',s=40)
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
    coefficients for an order-p filter. This version only works for tp=4,6,8.
    
    For tp=4,6 the analytical formulas in NR3 Sec. 13.10.1 are used. For tp>=8
    the coefficients tabulated in https://wavelets.pybytes.com/wavelet/db4/
    etc. are used - in this website these are called the "reconstruction
    low-pass filter coefficients".
    
    Parameters
    ----------
    tp : int in 2,4,6,8,10,20
        Number of coefficients returned.
    
    Returns
    -------
    cc : array (tp) of float
        Daubechies filter coefficients.
    """
    if tp==2:
        cc = np.zeros(2)
        d2d = sqrt(2)
        cc[:] = (1/d2d,
                 1/d2d)
        return cc
    if tp==4:
        cc = np.zeros(4)
        s3 = sqrt(3)
        d4d = 4*sqrt(2)
        cc[:] = ((1 + s3)/d4d,
                 (3 + s3)/d4d,
                 (3 - s3)/d4d,
                 (1 - s3)/d4d)
        return cc
    elif tp==6:
        cc = np.zeros(6)
        s10 = sqrt(10)
        s52 = sqrt(5 + 2*s10)
        d6d = 16*sqrt(2)
        cc[:] = ((1 + s10 + s52)/d6d,
                 (5 + s10 + 3*s52)/d6d,
                 (10 - 2*s10 + 2*s52)/d6d,
                 (10 - 2*s10 - 2*s52)/d6d,
                 (5 + s10 -3*s52)/d6d,
                 (1 + s10 - s52)/d6d)       
        return cc
    elif tp==8:
        cc = np.zeros(8)
        cc[:] = (0.23037781330885523,
                 0.7148465705525415,
                 0.6308807679295904,
                 -0.02798376941698385,
                 -0.18703481171888114,
                 0.030841381835986965,
                 0.032883011666982945,
                 -0.010597401784997278)
        return cc
    elif tp==10:
        cc = np.zeros(10)
        cc[:] = (0.160102397974125,
                 0.6038292697974729,
                 0.7243085284385744,
                 0.13842814590110342,
                 -0.24229488706619015,
                 -0.03224486958502952,
                 0.07757149384006515,
                 -0.006241490213011705,
                 -0.012580751999015526,
                 0.003335725285001549)
        return cc
    elif tp==20:
        cc = np.zeros(20)
        cc[:] = (0.026670057900950818,
                 0.18817680007762133,
                 0.5272011889309198,
                 0.6884590394525921,
                 0.2811723436604265,
                 -0.24984642432648865,
                 -0.19594627437659665,
                 0.12736934033574265,
                 0.09305736460380659,
                 -0.07139414716586077,
                 -0.02945753682194567,
                 0.03321267405893324,
                 0.0036065535669883944,
                 -0.010733175482979604,
                 0.0013953517469940798,
                 0.00199240529499085,
                 -0.0006858566950046825,
                 -0.0001164668549943862,
                 9.358867000108985e-05,
                 -1.326420300235487e-05)
        return cc
    else:
        raise Exception(f'Daubechies size {tp} is not among implemented'
                        ' values 2,4,6,8,10,20')

class Daub4(Wavelet):
    """Daubechies 4-coefficient wavelet, adapted from NR3 Sec. 13.10.2. Should
    give same results as Daubs(tp=4) if alternative centering is used in Daubs.
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
            ws[i+1] = cc0*a[n-2] + cc1*a[n-1] + cc2*a[0] + cc3*a[1]
            ws[i+nh+1] = cc3*a[n-2] - cc2*a[n-1] + cc1*a[0] - cc0*a[1]
        else:
            # Apply transpose filter.
            ws[0] = cc2*a[nh-1] + cc1*a[n-1] + cc0*a[0] + cc3*a[nh]
            ws[1] = cc3*a[nh-1] - cc0*a[n-1] + cc1*a[0] - cc2*a[nh]
            for i in range(nh-1):
                j = 2*(i+1)
                ws[j] = cc2*a[i] + cc1*a[i+nh] + cc0*a[i+1] + cc3*a[i+nh+1]
                ws[j+1] = cc3*a[i] - cc0*a[i+nh] + cc1*a[i+1] - cc2*a[i+nh+1]
        a[:n] = ws[:n]

class Daubs(Wavelet):
    """Daubechies tp-coefficient wavelet for tp=4,6... Adapted from NR3
    Sec. 13.10.2.
    
    Parameters
    ----------
    tp : int
        Number of coefficients 4,6... = 2*order of filter.  Allowed values
        are those allowed by daubscc.
    """
    def __init__(self,tp):
        self.tp = tp
        self.ccs = daubccs(tp)   # get and save D4 filter coefficients
        # Get reverse coefficients.
        self.ccrs = np.zeros_like(self.ccs)
        sig = -1.0
        for i in range(tp):
            self.ccrs[tp-1-i] = sig*self.ccs[i]
            sig = -sig
        # Standard centering.
        self.ioff = -(tp>>1)
        self.joff = self.ioff
        # Alternate centering, used by Daub4
        self.ioff = -2
        self.joff = -tp + 2

    def filt(self,a,n,forward):
        """Applies wavelet filter (if forward is true) or its transpose
        (otherwise) to a[0..n-1]. Called hierarchically by wt1 and wtn.
        """
        if n<4:
            return
        ws = np.zeros(n)
        nmod = self.tp*n       # a positive constant equal to zero mod n
        n1 = n-1               # mask of all bits, since n is a power of 2
        nh = n>>1
        if forward:
            # Apply filter.
            for ii,i in enumerate(range(0,n,2)):
                ni = i + 1 + nmod + self.ioff
                nj = i + 1 + nmod + self.joff
                for k in range(self.tp):
                    jf = n1 & (ni + k + 1)
                    jr = n1 & (nj + k + 1)
                    ws[ii] += self.ccs[k]*a[jf]
                    ws[ii+nh] += self.ccrs[k]*a[jr]
        else:
            # Apply transpose filter.
            for ii,i in enumerate(range(0,n,2)):
                ai = a[ii]
                ai1 = a[ii+nh]
                ni = i + 1 + nmod + self.ioff
                nj = i + 1 + nmod + self.joff
                for k in range(self.tp):
                    jf = n1 & (ni + k + 1)
                    jr = n1 & (nj + k + 1)
                    ws[jf] += self.ccs[k]*ai
                    ws[jr] += self.ccrs[k]*ai1
        a[:n] = ws[:n]

""" ******************** end of module wavl.py *************************** """