import numpy as np
import pandas as pd
import unittest

from fastEDM import edm
from helper import *


class TestCI(unittest.TestCase):
    def test_create_manifold(self):
        # Basic manifold creation [basicManifold]
        E = 2
        tau = 1
        p = 1
        t = [1, 2, 3, 4]
        x = [11, 12, 13, 14]

        # Basic manifold, no extras or dt
        res = edm(t, x, E=E, tau=tau, p=p, saveTargets=True, saveManifolds=True)
        true_library_mani = np.array([[12, 11]])
        true_prediction_mani = np.array([[13, 12], [14, 13]])

        true_targets = np.array([[14], [np.NAN]])
        print(true_targets)
        print(res["targets"])

        assert np.allclose(res["Ms"][0], true_library_mani)
        assert np.allclose(res["Mps"][0], true_prediction_mani)
        assert np.allclose(res["targets"], true_targets, equal_nan=True)
