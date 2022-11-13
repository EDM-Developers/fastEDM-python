from scipy.optimize import minimize
from scipy.stats import ks_2samp
import numpy as np
import math
import matplotlib.pyplot as plt

from fastEDM import edm

debug = True


def easy_edm(cause, effect, time = None, data = None, direction = "oneway", 
             verbosity = 1, showProgressBar = None, normalize = True):
    '''
    Simplified interface for EDM function
    ...
    Parameters
    ----------
    cause : str or list  
        The causal time series (as a string or a vector).
    effect : str or list 
        The effect time series (as a string or a vector).
    time : str or list 
        For non-regularly sampled time series, the sampling times must be supplied here.
    data : Pandas.DataFrame or None
        If a dataframe is supplied here, then the cause, effect & time
        arguments must be the column names of the relevant time series as strings.
    direction : str
        A string specifying whether we are checking a one directional causal effect or whether 
        to test the reverse direction at the same time (work in progress!).
    verbosity : int
        The level of detail in the output.
    showProgressBar : bool
        Whether or not to print out a progress bar during the computations.
    normalize : bool
        Whether to normalize the inputs before starting EDM.
    Returns
    ----------
    bool: Indicator for if evidence of causation was found.
    '''

    max_theta, num_thetas, theta_reps = 5, 100, 20 # !! Parameterise these values later
    convergence_method = "parametric"              # !! Parameterise these values later

    if not showProgressBar:
        showProgressBar = verbosity > 0

    # Convert time series to arrays (they can be supplied as columns of a dataframe).
    t, x, y = preprocess_inputs(data, cause, effect, time, verbosity, normalize)

    if (debug):
        print(f"\n=== Finding optimal E using simplex projection.")
        
    # Find optimal E (embedding dimension) of the causal variable using simplex projection
    E_best = find_embedding_dimension(t, x, verbosity, showProgressBar)

    if (debug):
        print(f"\n=== Testing for non-linearity using S-Map.")
        
    # Test for non-linearity using S-Map    
    test_nonlinearity(t, x, E_best, max_theta, num_thetas, theta_reps, verbosity, showProgressBar)
    
    if (debug):
        print(f"\n=== Testing for causality using CCM.")
        
    # Perform cross-mapping (CCM)
    res = cross_mapping(t, x, y, E_best, verbosity, showProgressBar)
    
    # Test for causality using CCM
    if convergence_method == "parametric":
        conv_test = test_convergence_monster
    elif convergence_method == "hypothesis":
        conv_test = test_convergence_monster # Replace this later
    else:
        conv_test = test_convergence_monster # Replace this later
    
    outcome = conv_test(res, data, cause, effect, verbosity)
    
    return outcome


def preprocess_inputs(data, cause, effect, time, verbosity, normalize):
    '''
    Convert time series to arrays (they can be supplied as columns of a dataframe).
    '''
    givenTimeSeriesNames = data is not None
    if givenTimeSeriesNames:
        if verbosity > 0:
            print("Pulling the time series from the supplied dataframe.")
        if cause not in data.columns:
            print(f"{cause} is not a column in the supplied dataframe.")
            exit(1)
        if effect not in data.columns:
            print(f"{effect} is not a column in the supplied dataframe.")
            exit(1)
        if time is not None and time not in data.columns:
            print(f"{time} is not a column in the supplied dataframe.")
            exit(1)
        x, y = data[cause], data[effect]
        t = data[time] if time else list(range(len(x)))
    else:
        if verbosity > 0:
            print("Using supplied time series vectors.")
        x, y = np.asarray(cause), np.asarray(effect)
        t = time if time else list(range(len(x)))

    if len(t) != len(x) or len(t) != len(y):
        print("Time series are not the same len .")
        exit(1)

    if (verbosity > 0):
        print(f"Number of observations is {len(t)}")

    if (normalize):
        if (verbosity > 0):
            print("Normalizing the supplied time series")
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        
    return t, x, y


def find_embedding_dimension(t, x, verbosity, showProgressBar):
    '''
    Find optimal E (embedding dimension) of the causal variable using simplex projection
    '''
    res = edm(t, x, E = list(range(3, 10 + 1)),
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
    
    return E_best


def test_nonlinearity(t, x, E_best, max_theta, num_thetas, theta_reps, verbosity, showProgressBar):
    '''
    Test for non-linearity using S-Map
    '''    
    theta_values = np.linspace(0, max_theta, 1 + num_thetas)

    # Calculate predictive accuracy over theta 0 to 'max_theta'
    res = edm(t, x, E = E_best, theta = theta_values, algorithm="smap", k=float("inf"),
              verbosity = 0, showProgressBar = showProgressBar)
    summary = res['summary']

    if (debug):
        print(f"Summary:\n{summary}")

    # Find optimal value of theta
    optIndex = summary['rho'].idxmax()
    optTheta = summary.iloc[optIndex]['theta']
    optRho   = round(summary.iloc[optIndex]['rho'], 5)

    if (verbosity > 0):
        print(f"Found optimal theta to be {optTheta}, with rho = {optRho}.")

    if (verbosity > 1):
        # Plot the rho-theta curve.
        plt.plot(summary["theta"], summary["rho"])
        plt.xlabel("theta")
        plt.ylabel("rho")
        plt.axvline(optTheta, ls="--", c="k")
        plt.show()

    # Kolmogorov-Smirnov test: optimal theta against theta = 0
    resBase = edm(t, x, E = E_best, theta = 0, numReps = theta_reps, k=20, algorithm="smap",
                  verbosity = 0, showProgressBar = showProgressBar)
    resOpt  = edm(t, x, E = E_best, theta = float(optTheta), numReps = theta_reps, k=20, algorithm="smap",
                  verbosity = 0, showProgressBar = showProgressBar)
        
    sampleBase, sampleOpt = resBase['stats']['rho'], resOpt['stats']['rho']
    if (debug):
        print(f"Theta=0:\n{sampleBase}")
        print(f"Theta>0:\n{sampleOpt}")
        print()
    
    ksTest = ks_2samp(sampleOpt, sampleBase, alternative='less')
    ksStat, ksPVal = round(ksTest.statistic, 5), round(ksTest.pvalue, 5)
    
    if (verbosity > 0):
        print(f"Found Kolmogorov-Smirnov test statistic to be {ksStat} with p-value={ksPVal}.")

    return


def cross_mapping(t, x, y, E_best, verbosity, showProgressBar):
    '''
    Find the maximum library size using S-map and this E selection
    '''
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
    return edm(t, y, x, E = E_best, library = libraries, algorithm = "smap", k = np.inf, 
              shuffle = True, verbosity = 0, showProgressBar = showProgressBar)


def test_convergence_monster(res, data, cause, effect, verbosity):
    '''
    Test for convergence using parametric test (Monster)
    '''
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
        ccmRho = round(monsterFit["rhoInfinity"], 2)
        print(f"The CCM final rho was {ccmRho}")

    if (monsterFit["rhoInfinity"] > 0.7):
        causalSummary = "Strong evidence"
    elif (monsterFit["rhoInfinity"] > 0.5):
        causalSummary = "Some evidence"
    else:
        causalSummary = "No evidence"

    givenTimeSeriesNames = data is not None
    if givenTimeSeriesNames:
        print(f"{causalSummary} of CCM causation from {cause} to {effect} found.")
    else:
        print(f"{causalSummary} of CCM causation found.")

    return monsterFit["rhoInfinity"] > 0.5
