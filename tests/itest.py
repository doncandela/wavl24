"""A test module for the wavl24 package.

itest.py simple tests that wavl24 can be imported.

"""
THIS_IS = 'itest.py 2/2/24 D. Candela'

from wavl24 import wavl

if __name__=='__main__':
    print(f'This is: {THIS_IS}')
    print(f'Using: {wavl.THIS_IS}')
