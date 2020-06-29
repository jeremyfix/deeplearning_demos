# coding: utf-8
"""
This scripts provides an interface to the bts library
"""

try:
    import bts
except ImportError:
    print("Warning: cannot import bts")


class BTS:

    def __init__(self):
        pass

    def __call__(self, ndimage):
        '''
        Process an image through the model
        and returns a Gray image
        '''
        return ndimage
