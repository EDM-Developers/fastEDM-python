import numpy as np
import numpy.random as rnd
import pandas as pd
import unittest

from fastEDM import edm, create_manifold_and_targets
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

    def test_save_targets(self):
        # Create a sine-like time series with noise
        # (used as an example in Strasbourg talk).
        rnd.seed(42)

        f = 0.137

        a = 2 * np.cos(2 * np.pi * f)
        b = -1

        N = 100
        t = np.arange(N).astype(float)
        x = np.zeros(N)
        x[1] = 1.0

        for i in range(2, N):
            x[i] = a * x[i - 1] + b * x[i - 2] + rnd.normal(0, 0.1)

        x = x.round(2)

        E = 3
        tau = 1
        p = 1

        mani, targets = create_manifold_and_targets(t, x, E=E, tau=tau, p=p)

        assert type(mani) == np.ndarray
        assert type(targets) == np.ndarray

        mani = mani[:, ::-1]

        assert len(x) == 100
        assert len(mani) == 97

        assert targets[0] == x[3]
        assert targets[1] == x[4]
        assert targets[2] == x[5]
        assert targets[3] == x[6]
