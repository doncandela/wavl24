"""A test program for the wavl24 package.

wavltests.py can make various plots using Daubechies wavelets.
"""
THIS_IS = 'wavltests.py 2/8/24 D. Candela'

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from wavl24 import wavl

def nr3plots():
    """Make plots of basis wavelets as in Numerical Recipes 3rd Ed.
    """
    # Plot like NR3 Fig. 13.10.1 top: Daubechies p=2, basis wavelet 4.
    wavelet = wavl.getwavelet(wltype='daub',p=2)
    a = wavl.bwavs(1024,(4,),wavelet)
    plt.plot(a)
    plt.title('Daubechies p=2, basis wavelet 4')
    plt.show()
    # Plot like NR3 Fig. 13.10.1 top: Daubechies p=10, basis vector 21.
    wavelet = wavl.getwavelet(wltype='daub',p=10)
    a = wavl.bwavs(1024,(21,),wavelet)
    plt.plot(a)
    plt.title('Daubechies p=10, basis wavelet 21')
    plt.show()
    # Plot like NR3 Fig. 13.10.2 top: Daubechies p=2, basis wavelets 9+57.
    wavelet = wavl.getwavelet(wltype='daub',p=2)
    a = wavl.bwavs(1024,(9,57),wavelet)
    plt.plot(a)
    plt.title('Daubechies p=2, basis wavelets 9+57')
    plt.show()

def plot32(wltype='daub',p=2):
    """Plot first 32 basis vectors with N=128, compare with same-numbered
    basis vectors with N=1024.
    
    Parameters
    ----------
    wltype,t
        Type and order of wavelet filter as passed to wavl.getwavelet (see).
    """
    wavelet = wavl.getwavelet(wltype=wltype,p=p)
    # x arrays for the two different length vectors.
    x128 = np.arange(0.,1.,1/128)
    x1024 = np.arange(0.,1.,1/1024)
    for nb in range(32):
        a128 = np.zeros(128)
        a128[nb] = 1.0
        wavl.wt1(a128,wavelet,forward=False)
        # Scale to 1024 amplitude.
        a128 *= sqrt(128/1024)

        a1024 = np.zeros(1024)
        a1024[nb] = 1.0
        wavl.wt1(a1024,wavelet,forward=False)

        plt.title(f'p={p} nb={nb}')
        plt.plot(x128,a128,label='N=128')
        plt.plot(x1024,a1024,label='N=1024')
        plt.legend()
        plt.show()

def showrand(nn,nnshow=None,dots=()):
    """Tests dwt display function wavl.showdwt with a dwt of size nn
    populated with random numbers.  Supply nnshow<nn to truncate
    display to nnshow elements.
    """
    rng = np.random.default_rng(2024)   # get seeded random number gen
    dwt = rng.normal(size=nn)            # use gaussian-distributed elements
    wavl.showdwt(dwt,nnshow,dots=dots)
    plt.title(f'Random dwt with N={nn}, Nshow={nnshow}')
    plt.show()
    
def wt_iwt(nn,ns,wltype='daub',p=2):
    """Gets and plots the sum of one or more basis wavelets, then does the wt
    on this sum and displays the result -- should have all weight in the
    chosen basis vectors.

    Parameters
    ----------
    nn : int
        Size of data, must be power of 2.
    ns : int,int...
        Which basis functions to use, each 0...nn-1.
    wltype,t
        Type and order of wavelet filter as passed to wavl.getwavelet (see).
    """
    wavelet = wavl.getwavelet(wltype=wltype,p=p)
    # Make normalized data with weight in specified basis functions.
    a = np.zeros(nn)
    wgt = 1/sqrt(len(ns))
    for n in ns:
        a[n] = wgt
    # Do iwt to get sum of corresponding wavelets, plot result.
    wavl.wt1(a,wavelet,forward=False)
    plt.plot(a)
    plt.title(f'p = {p}, sum of basis vectors {ns}')
    plt.show()
    # Do iwt, display.
    wavl.wt1(a,wavelet)
    wavl.showdwt(a,dots=ns)
    plt.title(f'Sum of basis vectors {ns}')
    # plt.colorbar()
    plt.show()

"""************ RUN TEST FUNCTIONS *****************"""
if __name__=='__main__':
    print(f'This is: {THIS_IS}')
    print(f'Using: {wavl.THIS_IS}')   
    # nr3plots()    
    # plot32(p=3)
    # showrand(nn=1024,nnshow=128,dots=(0,7,122))
    wt_iwt(1024,(18,),p=4)