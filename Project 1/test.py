#!/usr/bin/env python

import unittest
import numpy as np

from project1 import *


class TestSolution(unittest.TestCase):
    
    def test_extrapolate_depth(self):
    
        depth = extrapolate_depth('Nechelik_Data.csv', 54, 44)

        ans = np.array([[ 6830,  6812,  6797,  6786, 6780,  6779,  6785,  6796, 6818,  6842],
           [ 6847,  6827,  6812,  6801, 6794,  6792,  6797,  6815, 6840,  6867], 
           [ 6863,  6845,  6830,  6820, 6812,  6812,  6821,  6838, 6862,  6889],
           [ 6883,  6864,  6852,  6839, 6836,  6836,  6845,  6861, 6881,  6906],
           [ 6903,  6887,  6871,  6863, 6859,  6859,  6867,  6881, 6903,  6926],
           [ 6926,  6910,  6893,  6886, 6881,  6881,  6888,  6903, 6922,  6945]])

        np.testing.assert_allclose(depth[17:23,25:35], ans, atol=30)
    
    def test_nans(self):
    
        depth = extrapolate_depth('Nechelik_Data.csv', 54, 44)

        ans = np.array([[ np.nan,  np.nan],
                        [ np.nan,  np.nan],
                        [ np.nan,  np.nan] ])

        np.testing.assert_allclose(depth[:3,:2], ans)

if __name__ == '__main__':
        unittest.main()
