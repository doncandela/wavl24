"""A test program for the wavl24 package.

wavltests.py can make various plots using Daubechies wavelets.
"""
THIS_IS = 'wavltests.py 2/6/24 D. Candela'

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from wavl24 import wavl

def nr3plots(tp=None):
    """Make plots of basis wavelets as in Numerical Recipes 3rd Ed.
    
    Parameters
    ----------
    tp : None or int
        2*order = 4,6...  using Daubs, or None to use Daub4.
    """
    wavelet = wavl.getfilter(tp)
    # Plot like NR3 Fig. 13.10.1 top: wavelet 4 in 1024-length vectors.
    a = np.zeros(1024)
    a[4] = 1.0
    wavl.wt1(a,wavelet,forward=False)
    plt.plot(a)
    plt.show()
    # Plot like NR3 Fig. 13.10.2 top: wavelets 9 + 57 in 1024-length vectors.
    a = np.zeros(1024)
    a[9] = 1.0
    a[57] = 1.0
    wavl.wt1(a,wavelet,forward=False)
    plt.plot(a)
    plt.show()
    
def plot32(tp=None):
    """Plot all 32 basis vectors with N=32, compare with same-numbered
    basis vectors with N=1024.
    
    Parameters
    ----------
    tp : None or int
        2*order = 4,6...  using Daubs, or None to use Daub4.
    """
    wavelet = wavl.getfilter(tp)
    # x arrays for the two different length vectors.
    x32 = np.arange(0.,1.,1/32)
    x1024 = np.arange(0.,1.,1/1024)
    for nb in range(32):
        a32 = np.zeros(32)
        a32[nb] = 1.0
        wavl.wt1(a32,wavelet,forward=False)
        # Scale to 1024 amplitude.
        a32 *= sqrt(32/1024)

        a1024 = np.zeros(1024)
        a1024[nb] = 1.0
        wavl.wt1(a1024,wavelet,forward=False)

        plt.title(f'nb={nb}')
        plt.plot(x32,a32,label='N=32')
        plt.plot(x1024,a1024,label='N=1024')
        plt.legend()
        plt.show()

def showrand(nn,nnshow=None):
    """Tests dwt display function wavl.showdwt with a dwt of size nn
    populated with random numbers.  Supply nnshow<nn to truncate
    display to nnshow elements.
    """
    rng = np.random.default_rng(2024)   # get seeded random number gen
    dwt = rng.normal(size=nn)            # use gaussian-distributed elements
    wavl.showdwt(dwt,nnshow)
    plt.title(f'Random dwt with N={nn}, Nshow={nnshow}')
    plt.show()
    
def wt_iwt(nn,ns,tp=None):
    """Gets the sum of one or more wavelets by doing the iwt on an array with
    corresponding nonzero elements, plots the result, then does the wt and
    displays the result -- should have all weight in the chosen basis vectors.

    Parameters
    ----------
    nn : int
        Size of data, must be power of 2.
    n : int,int...
        Which basis functions to use, each 0...nn-1.
    tp : None or int
        2*order = 4,6...  using Daubs, or None to use Daub4.
    """
    wavelet = wavl.getfilter(tp)
    # Make normalized data with weight in specified basis functions.
    a = np.zeros(nn)
    wgt = 1/sqrt(len(ns))
    for n in ns:
        a[n] = wgt
    # Do iwt to get sum of corresponding wavelets, plot result.
    wavl.wt1(a,wavelet,forward=False)
    plt.plot(a)
    plt.title(f'Sum of basis vectors {ns}')
    plt.show()
    # Do iwt, display.
    wavl.wt1(a,wavelet)
    wavl.showdwt(a)
    plt.title(f'Sum of basis vectors {ns}')
    plt.colorbar()
    plt.show()


"""************ RUN TEST FUNCTIONS *****************"""
if __name__=='__main__':
    print(f'This is: {THIS_IS}')
    print(f'Using: {wavl.THIS_IS}')
    
    # nr3plots()    
    # plot32()
    # showrand(nn=1024,nnshow=128)
    wt_iwt(1024,(33,40,99))