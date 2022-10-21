import numpy as np
import pandas as pd
import unittest

from fastEDM import easy_edm
from helper import logistic_map


class TestEasyEdm(unittest.TestCase):
    def test_logistic_map(self):
        x, y = logistic_map(4)
        assert np.allclose(x, [0.2000000, 0.5760000, 0.8601754, 0.4266398])
        assert np.allclose(y, [0.4000000, 0.8728000, 0.2174529, 0.5667110])
        
    def test_simple_manifold(self):
        obs = 500
        x, y = logistic_map(obs)

        df = pd.DataFrame({"x": x, "y":y})

        # Check that passing the data via a dataframe works.
        xCCMCausesY = easy_edm("x", "y", data = df)
        assert xCCMCausesY == True

        # yCCMCausesX = easy_edm("y", "x", data = df)
        # assert yCCMCausesX == True

        # Check that passing the raw data is also fine.
        xCCMCausesY = easy_edm(x, y)
        assert xCCMCausesY == True
        
    def test_chicago_dataset(self):
        url = "https://github.com/EDM-Developers/fastEDM/raw/master/vignettes/chicago.csv"
        chicago = pd.read_csv(url)
        
        crimeCCMCausesTemp = easy_edm("Crime", "Temperature", data = chicago)
        assert crimeCCMCausesTemp == False

        tempCCMCausesCrime = easy_edm("Temperature", "Crime", data = chicago)
        assert tempCCMCausesCrime == True

        # Check that the results still hold up if we don't normalize the inputs
        crimeCCMCausesTemp = easy_edm("Crime", "Temperature", data = chicago, normalize = False)
        assert crimeCCMCausesTemp == False

        tempCCMCausesCrime = easy_edm("Temperature", "Crime", data = chicago, normalize = False)
        assert tempCCMCausesCrime == True
    