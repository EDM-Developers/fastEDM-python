from fastEDM import easy_edm
import pandas as pd

url = "https://github.com/EDM-Developers/fastEDM/raw/master/vignettes/chicago.csv"
chicago = pd.read_csv(url)
chicago["Crime"] = chicago["Crime"].diff()

crimeCCMCausesTemp = easy_edm("Crime", "Temperature", data=chicago, verbosity=0)
#> No evidence of CCM causation from Crime to Temperature found.
tempCCMCausesCrime = easy_edm("Temperature", "Crime", data=chicago, verbosity=0)
#> Some evidence of CCM causation from Temperature to Crime found.
