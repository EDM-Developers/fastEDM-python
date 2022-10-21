from scipy.optimize import minimize
import numpy as np
import math

from fastEDM import edm


#' easy_edm
#'
#' @param cause The causal time series (as a string or a vector).
#'
#' @param effect The effect time series (as a string or a vector).
#'
#' @param time For non-regularly sampled time series, the sampling times
#' must be supplied here (as a string or a vector).
#'
#' @param data If a dataframe is supplied here, then the cause, effect & time
#' arguments must be the column names of the relevant time series as strings.
#'
#' @param direction A string specifying whether we are checking a one
#' directional causal effect or whether to test the reverse direction at the
#' same time (work in progress!).
#'
#' @param verbosity The level of detail in the output.
#'
#' @param showProgressBar Whether or not to print out a progress bar during the computations.
#'
#' @param normalize Whether to normalize the inputs before starting EDM.
#'
#' @returns A Boolean indicating that evidence of causation was found.
#' @export
#' @example man/chicago-easy-edm-example.R
#
def easy_edm(cause, effect, time = None, data = None, direction = "oneway", 
             verbosity = 1, showProgressBar = None, normalize = True):

    if not showProgressBar:
        showProgressBar = verbosity > 0

    # If doing a rigorous check, begin by seeing if the cause & effect
    # variable appear to be non-linear dynamical system outputs, or just
    # random noise.
    # TODO

    # First find out the embedding dimension of the causal variable
    givenTimeSeriesNames = data is not None
    if givenTimeSeriesNames:
        if verbosity > 0:
            print("Pulling the time series from the supplied dataframe.")
        if cause not in data.columns:
            print(f"{cause} is not a column in the supplied dataframe.")
            return 1
        if effect not in data.columns:
            print(f"{effect} is not a column in the supplied dataframe.")
            return 1
        if time is not None and time not in data.columns:
            print(f"{time} is not a column in the supplied dataframe.")
            return 1
        x, y = data[cause], data[effect]
        t = data[time] if time else list(range(len(x)))
    else:
        if verbosity > 0:
            print("Using supplied time series vectors.")
        x, y = np.asarray(cause), np.asarray(effect)
        t = time if time else list(range(len(x)))

    if len(t) != len(x) or len(t) != len(y):
        print("Time series are not the same len .")
        return 1

    if (verbosity > 0):
        print(f"Number of observations is {len(t)}")

    if (normalize):
        if (verbosity > 0):
            print("Normalizing the supplied time series")
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

    res = edm(t, y, E = list(range(3,10 + 1)), 
              verbosity = 0, showProgressBar = showProgressBar)

    if res["rc"] > 0:
        print("Search for optimal embedding dimension failed.")
        return 2

    if res["summary"]["rho"] is None or len(res["summary"]["rho"]) == 0:
        print("Search for optimal embedding dimension failed (2).")
        return 3

    idx_max_rho = np.argmax(res["summary"]["rho"])
    E_best = int(res["summary"]["E"][idx_max_rho])

    if (verbosity > 0):
        print(f"Found optimal embedding dimension E to be {E_best}.")

    # Find the maximum library size using S-map and this E selection
    res = edm(t, y, E = E_best, algorithm = "smap", 
              full = True, saveManifolds = True, 
              verbosity = 0, showProgressBar = showProgressBar)
    libraryMax = len(res["Ms"][0])

    if (verbosity > 0):
        print(f"The maximum library size we can use is {libraryMax}.")

    # Set up a grid of library sizes to run the cross-mapping over.
    if (libraryMax >= 500):
        libraryStart = 100
    else:
        libraryStart = 10
    
    step = (libraryMax - libraryStart) / 25
    libraries = [math.ceil(libraryStart + step * k) for k in range(25)]

    # Next run the convergent cross-mapping (CCM), using the effect to predict the cause.
    res = edm(t, y, x, E = E_best, library = libraries, algorithm = "smap", k = np.inf, 
              shuffle = True, verbosity = 0, showProgressBar = showProgressBar)

    # Make some rough guesses for the Monster exponential fit coefficients
    ccmRes = res["summary"]
    firstLibrary = np.asarray(ccmRes["library"][0])
    firstRho = np.asarray(ccmRes["rho"])[0]
    finalRho = np.asarray(ccmRes["rho"])[-1]

    gammaGuess = 0.001
    rhoInfinityGuess = finalRho
    alphaGuess = (firstRho - rhoInfinityGuess) / math.exp(-gammaGuess * firstLibrary)

    monsterFitStart = { 
        "alpha": alphaGuess, 
        "gamma": gammaGuess, 
        "rhoInfinity": rhoInfinityGuess 
    }
    
    mArgs = ccmRes.filter(['alpha', 'gamma', 'library', 'rhoInfinity'])
    def mTarget(alpha, gamma, library, rhoInfinity):
        return alpha * np.exp(-gamma * library) + rhoInfinity

    try:
        mFit = minimize(mTarget, initial_guess = monsterFitStart, args = mArgs)
        monsterFit = { 
            "alpha": mFit["alpha"], 
            "gamma": mFit["gamma"], 
            "rhoInfinity": mFit["rhoInfinity"] 
        }
    except Exception as e:
        if (verbosity > 0):
            print("Couldn't fit an exponential curve to the rho-L values.")
            print("This may be a sign that these EDM results are not very reliable.")
        monsterFit = {
            "alpha": None, "gamma": None, "rhoInfinity": finalRho
        }

    if (verbosity > 1):
        ccmAlpha = round(monsterFit["alpha"], 2)
        ccmGamma = round(monsterFit["gamma"], 2)
        ccmRho = round(monsterFit["rhoInfinity"], 2)
        print(f"The CCM fit is (alpha, gamma, rhoInfinity) = ({ccmAlpha}, {ccmGamma}, {ccmRho}).")
    elif (verbosity == 1):
        ccmRho = {round(monsterFit["rhoInfinity"], 2)}
        print(f"The CCM final rho was {ccmRho}")

    if (monsterFit["rhoInfinity"] > 0.7):
        causalSummary = "Strong evidence"
    elif (monsterFit["rhoInfinity"] > 0.5):
        causalSummary = "Some evidence"
    else:
        causalSummary = "No evidence"

    if givenTimeSeriesNames:
        print(f"{causalSummary} of CCM causation from {cause} to {effect} found.")
    else:
        print(f"{causalSummary} of CCM causation found.")

    return monsterFit["rhoInfinity"] > 0.5
