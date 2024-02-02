"""A test program for the wavl24 package.

itest.py simple tests that wavl24 can be imported.
"""
THIS_IS = 'itest.py 2/2/24 D. Candela'

from wavl24 import wavl

if __name__=='__main__':
    print(f'This is: {THIS_IS}')
    print(f'Using: {wavl.THIS_IS}')
    
    # Print Daub4 coefficients.
    print(wavl.D4C0)
    print(wavl.D4C1)
    print(wavl.D4C2)
    print(wavl.D4C3)
