"""A test program for the wavl24 package.

plotd4.py plots some Daubechies 4-coefficient wavelets
"""
THIS_IS = 'plotd4.py 2/2/24 D. Candela'

import numpy as np
import matplotlib.pyplot as plt
from wavl24 import wavl

if __name__=='__main__':
    print(f'This is: {THIS_IS}')
    print(f'Using: {wavl.THIS_IS}')
    
    # Plot like NR3 Fig. 13.10.1 top: wavelet 4 in 1024-length vectors.
    a = np.zeros(1024)
    a[4] = 1.0
    wavl.wt1(a,wavl.Daub4,forward=False)
    plt.plot(a)
    plt.show()

    # Plot like NR3 Fig. 13.10.2 top: wavelets 9 + 57 in 1024-length vectors.
    a = np.zeros(1024)
    a[9] = 1.0
    a[57] = 1.0
    wavl.wt1(a,wavl.Daub4,forward=False)
    plt.plot(a)
    plt.show()
