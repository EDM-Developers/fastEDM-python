import numpy as np
import pandas as pd
import unittest

from fastEDM import easy_edm
from helper import logistic_map

VERBOSITY = 0
NUM_THREADS = 4

class TestEasyEdm(unittest.TestCase):
    def test_logistic_map(self):
        x, y = logistic_map(4)
        assert np.allclose(x, [0.2000000, 0.5760000, 0.8601754, 0.4266398])
        assert np.allclose(y, [0.4000000, 0.8728000, 0.2174529, 0.5667110])

    def test_simple_manifold(self):
        obs = 500
        x, y = logistic_map(obs)

        df = pd.DataFrame({"x": x, "y": y})

        # Check that passing the data via a dataframe works.
        print("\n>>> Test 1")
        xCCMCausesY = easy_edm("x", "y", data=df, verbosity=VERBOSITY)
        assert xCCMCausesY == "Strong evidence"

        # yCCMCausesX = easy_edm("y", "x", data = df)
        # assert yCCMCausesX == "Strong evidence"

        # Check that passing the raw data is also fine.
        print("\n>>> Test 2")
        xCCMCausesY = easy_edm(x, y, verbosity=VERBOSITY)
        assert xCCMCausesY == "Strong evidence"

        # Check that larger values of verbosity work.
        # N.B. For verbosity > 2, we create a plot, which
        # will hang the tests if we don't close it.
        print("\n>>> Test 3")
        xCCMCausesY = easy_edm(x, y, verbosity=2)
        assert xCCMCausesY == "Strong evidence"

    def test_chicago_dataset(self):
        url = "https://github.com/EDM-Developers/fastEDM-r/raw/main/vignettes/chicago.csv"
        chicago = pd.read_csv(url)

        print("\n>>> Test 1")
        crimeCCMCausesTemp = easy_edm("Crime", "Temperature", data=chicago, verbosity=VERBOSITY, numThreads=NUM_THREADS)
        assert crimeCCMCausesTemp == "No evidence"

        print("\n>>> Test 2")
        tempCCMCausesCrime = easy_edm("Temperature", "Crime", data=chicago, verbosity=VERBOSITY, numThreads=NUM_THREADS)
        assert tempCCMCausesCrime != "No evidence"

        # Check that the results still hold up if we don't normalize the inputs
        print("\n>>> Test 3")
        crimeCCMCausesTemp = easy_edm("Crime", "Temperature", data=chicago, normalize=False, verbosity=VERBOSITY, numThreads=NUM_THREADS)
        assert crimeCCMCausesTemp == "No evidence"

        print("\n>>> Test 4")
        tempCCMCausesCrime = easy_edm("Temperature", "Crime", data=chicago, normalize=False, verbosity=VERBOSITY, numThreads=NUM_THREADS)
        assert tempCCMCausesCrime != "No evidence"
