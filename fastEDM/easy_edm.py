from scipy.optimize import minimize
from scipy.stats import ks_2samp
import numpy as np
import math
import matplotlib.pyplot as plt

from fastEDM import edm

DEBUG = False


def easy_edm(cause, effect, time = None, data = None, direction = "oneway", 
             verbosity = 1, showProgressBar = None, normalize = True):
    '''
    Simplified interface for EDM function
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    cause : str or list  
        The causal time series (as a string or a list).
    effect : str or list 
        The effect time series (as a string or a list).
    time : str or list 
        For non-regularly sampled time series, the sampling times must be supplied here.
    data : Pandas.DataFrame or None
        If a dataframe is supplied here, then the cause, effect & time
        arguments must be the column names of the relevant time series as strings.
    direction : str
        A string specifying whether we are checking a one directional causal effect or whether 
        to test the reverse direction at the same time (work in progress!).
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether or not to print out a progress bar during the computations.
    normalize : bool
        Whether to normalize the inputs before starting EDM.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    result : str
        Strength of evidence for causation. Either 'Strong evidence', 'Some evidence', 
        or 'No evidence'.
    '''

    print("While the 'edm' functionality is well-tested and ready for use, '"
          "this 'easy_edm automated analysis is still a work-in-progress.")

    # Development parameters, remove later!
    convergence_method = "quantile"
    max_theta = 5
    num_thetas = 100
    theta_reps = 20
    maxLag = 5
    
    if not showProgressBar:
        showProgressBar = verbosity > 0

    if verbosity > 1:
        print(f"=== Processing inputs and extracting relevant data.")

    # Convert time series to arrays (they can be supplied as columns of a dataframe).
    t, x, y = preprocess_inputs(data, cause, effect, time, verbosity, normalize)

    if verbosity > 1:
        print(f"=== Finding optimal E using simplex projection.")
        
    # Find optimal E (embedding dimension) of the causal variable using simplex projection
    E_best = find_embedding_dimension(t, x, verbosity, showProgressBar)
    
    if verbosity > 1:
        print(f"=== Testing for non-linearity using S-Map.")
    
    # Test for non-linearity using S-Map    
    optTheta, isNonLinear = test_nonlinearity(t, x, E_best, max_theta, num_thetas, theta_reps, 
                                              verbosity, showProgressBar)
    
    if verbosity > 1:
        print(f"=== Testing for delay effect of x on y.")
        
    # Lags the y (effect) time series by the optimal value or differences the series if it was linear
    yOpt = get_optimal_effect(t, x, y, E_best, verbosity, showProgressBar, isNonLinear, optTheta, maxLag)

    if verbosity > 1:
        print(f"=== Finding maximum library size.")
        
    # Get max library size
    libraryMax = get_max_library(t, x, yOpt, E_best, verbosity, showProgressBar)

    if verbosity > 1:
        print(f"=== Testing for causality using '{convergence_method}' method.")
        
    # Test for causality using CCM
    if convergence_method == "parametric":
        # Perform cross-mapping (CCM)
        result = test_convergence_monster(t, x, yOpt, E_best, libraryMax, optTheta, verbosity, showProgressBar)
    elif convergence_method == "hypothesis":
        result = test_convergence_monster # Replace this later
    elif convergence_method == "quantile":
        result = test_convergence_dist(t, x, yOpt, E_best, libraryMax, optTheta, verbosity, showProgressBar)
    else:
        raise ValueError("Invalid convergence method selected")

    if verbosity > 1:
        print(f"=== Results")
    
    givenTimeSeriesNames = data is not None
    if givenTimeSeriesNames:
        print(f"{result} of CCM causation from {cause} to {effect} found.")
    else:
        print(f"{result} of CCM causation found.")

    return result


def preprocess_inputs(data, cause, effect, time, verbosity, normalize): 
    '''
    Convert time series to arrays (they can be supplied as columns of a dataframe).
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    data : Pandas.DataFrame or None
        If a dataframe is supplied here, then the cause, effect & time
        arguments must be the column names of the relevant time series as strings.
    cause : str or list  
        The causal time series (as a string or a list).
    effect : str or list 
        The effect time series (as a string or a list).
    time : str or list 
        For non-regularly sampled time series, the sampling times must be supplied here 
        as a string or a list.
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    normalize : bool
        Whether to normalize the inputs before starting EDM.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    tuple(t, x, y)
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    y : np.ndarray
        The effect time series.
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
        x, y = np.array(data[cause], copy=True), np.array(data[effect], copy=True) 
        t = np.array(data[time], copy=True) if time else np.array(range(len(x)))
    else:
        if verbosity > 0:
            print("Using supplied time series lists.")
        x, y = np.array(cause, copy=True), np.array(effect, copy=True)
        t = np.array(time, copy=True) if time else np.array(range(len(x)))

    if len(t) != len(x) or len(t) != len(y):
        print("Time series are not the same len .")
        exit(1)

    if (verbosity > 0):
        print(f"Number of observations is {len(t)}")

    if (normalize):
        if (verbosity > 0):
            print("Normalizing the supplied time series")
        # Skips NaN values when calculatin mean and std dev
        x = (x - np.nanmean(x)) / np.nanstd(x)
        y = (y - np.nanmean(y)) / np.nanstd(y)
                
    return t, x, y


def find_embedding_dimension(t, x, verbosity, showProgressBar):
    '''
    Find optimal E (embedding dimension) of the causal variable using simplex projection
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether to print the progress bar for EDM calls.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    E_best : int
        Optimal embedding dimension of the causal variable
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
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    E_best : int
        Optimal embedding dimension of the causal variable
    max_theta : int
        Maximum theta value to use for testing nonlinearity.
    theta_reps :
        Number of reps to use for Kolmogorov-Smirnov test
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether to print the progress bar for EDM calls.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    (optTheta, isNonLinear)
    optTheta : int
        Optimal theta value found during nonlinearity testing.
    isNonLinear : bool
        Whether the system passes the nonlinearity test.
    ------------------------------------------------------------------------------------------
    '''
    
    theta_values = np.linspace(0, max_theta, 1 + num_thetas)

    # Calculate predictive accuracy over theta 0 to 'max_theta'
    res = edm(t, x, E = E_best, theta = theta_values, algorithm="smap", k=float("inf"),
              verbosity = 0, showProgressBar = showProgressBar)
    summary = res['summary']

    if (DEBUG):
        print(f"Summary:\n{summary}")

    # Find optimal value of theta
    optIndex = summary['rho'].idxmax()
    optTheta = float(summary.iloc[optIndex]['theta'])
    optRho   = round(summary.iloc[optIndex]['rho'], 5)

    if (verbosity > 0):
        print(f"Found optimal theta to be {round(optTheta, 3)}, with rho = {optRho}.")

    if (verbosity > 2):
        # Plot the rho-theta curve.
        plt.plot(summary["theta"], summary["rho"])
        plt.xlabel("theta")
        plt.ylabel("rho")
        plt.axvline(optTheta, ls="--", c="k")
        plt.show()

    # Kolmogorov-Smirnov test: optimal theta against theta = 0
    resBase = edm(t, x, E = E_best, theta = 0, numReps = theta_reps, k=20, algorithm="smap",
                  verbosity = 0, showProgressBar = showProgressBar)
    resOpt  = edm(t, x, E = E_best, theta = optTheta, numReps = theta_reps, k=20, algorithm="smap",
                  verbosity = 0, showProgressBar = showProgressBar)
        
    sampleBase, sampleOpt = resBase['stats']['rho'], resOpt['stats']['rho']
    
    if (DEBUG):
        print(f"Theta=0:\n{resBase['stats']}")
        print(f"Theta>0:\n{resOpt['stats']}\n")
    
    ksTest = ks_2samp(sampleOpt, sampleBase, alternative='less')
    ksStat, ksPVal = round(ksTest.statistic, 5), round(ksTest.pvalue, 5)
    
    if (verbosity > 0):
        print(f"Found Kolmogorov-Smirnov test statistic to be {ksStat} with p-value={ksPVal}.")

    isNonLinear = ksPVal < 0.05
    return optTheta, isNonLinear


def tslag(t, x, lag=1, dt=1):
    '''
    Helper: Lags a given time series
    '''
    t = np.asarray(t)
    x = np.asarray(x)
    l_x = np.full(len(t), np.nan)
    for i in range(len(t)):
        lagged_t = t[i] - lag * dt
        if not np.isnan(lagged_t) and lagged_t in t:
            l_x[i] = x[t == lagged_t]
            
    return l_x


def get_optimal_effect(t, x, y, E_best, verbosity, showProgressBar, isNonLinear, theta=1, maxLag=5):
    '''
    Find optimal lag for the y (effect) time series
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    y : np.ndarray
        The effect time series.
    E_best : int
        Optimal embedding dimension of the causal variable
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether to print the progress bar for EDM calls.
    isNonLinear : bool
        Whether the system passes the nonlinearity test.
    theta : int
        Theta value to use for EDM calls.
    maxLag : int
        Maximum range of lag values (i.e. will test all t s.t. -maxLag <= t <= maxLag).
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    yOpt : np.ndarray
        The modified effect time series. Either lagged by optimal lagging value or differenced
        if the time series does not pass the nonlinearity test.
    ------------------------------------------------------------------------------------------
    '''
    
    rhos = {}
    for i in range(-maxLag, maxLag + 1):
        res = edm(t, x, tslag(t, y, i), E = E_best, theta = theta, 
                  algorithm="simplex", k=float("inf"), verbosity = 0, showProgressBar = showProgressBar)
        rhos[i] = float(res['summary']['rho'])

    optLag = max(rhos, key=rhos.get)

    if (DEBUG):
        print("Lagged Rhos:")
        for k, v in rhos.items():
            print("Lag " + ("+" if k >= 0 else "") + str(k) + ": " + str(round(v, 5)))
    
    if (verbosity > 0): 
        print(f'Found optimal lag to be {optLag} with rho={round(rhos[optLag], 5)}')
    
    # If retrocausality is spotted, default to best positive lag and print warning
    invalidLag = optLag < 0
    if invalidLag:
        validRhos = {k: v for k, v in rhos if k >= 0}
        optLag = max(validRhos, key=validRhos.get)
        if (verbosity > 0):
            print(f'This may indicate retrocausality, using alternate lag of {optLag} \
                    with rho={round(rhos[optLag], 5)}')
        
    yLag = tslag(t, y, optLag)
    if isNonLinear:     
        # Transform y and data to match optimal lagged series
        yOpt = yLag
        if (verbosity > 0):
            print(f"Lagging time series using optimal lag of {optLag}")
    else:
        # Difference y and data by optimal lagged series        
        yOpt = y - yLag
        if verbosity > 0:
            print(f'Differencing time series due to failed nonlinearity test (lag={optLag})')
    
    return yOpt


def get_max_library(t, x, y, E_best, verbosity, showProgressBar):
    '''
    Finds the maximum library size
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    y : np.ndarray
        The effect time series.
    E_best : int
        Optimal embedding dimension of the causal variable
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether to print the progress bar for EDM calls.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    libraryMax : int
        Maximum library size
    ------------------------------------------------------------------------------------------
    '''
    
    res = edm(t, y, E = E_best, algorithm = "simplex", 
              full = True, saveManifolds = True, 
              verbosity = 0, showProgressBar = showProgressBar)
    libraryMax = len(res["Ms"][0])
    
    if verbosity > 0:
        print(f'Found maximum library size of {libraryMax}')
    return libraryMax


def cross_mapping(t, x, y, E_best, libraryMax, theta, verbosity, showProgressBar):
    '''
    Perform convergent cross mapping, using the effect to predict the cause.
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    y : np.ndarray
        The effect time series.
    E_best : int
        Optimal embedding dimension of the causal variable
    libraryMax : int
        Maximum library size
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether to print the progress bar for EDM calls.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    ccm : EDM output
        EDM output for the convergent cross mapping using the effect to predict the cause.
    ------------------------------------------------------------------------------------------
    '''
    
    res = edm(t, y, E = E_best, algorithm = "simplex", 
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
    ccm = edm(t, y, x, E = E_best, library = libraries, theta = theta, 
                           algorithm = "simplex", k = np.inf, shuffle = True, verbosity = 0, 
                           showProgressBar = showProgressBar)
    return ccm


        
def test_convergence_monster(t, x, y, E_best, libraryMax, theta, verbosity, showProgressBar):
    '''
    Test for convergence using parametric test (Monster)
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    y : np.ndarray
        The effect time series.
    E_best : int
        Optimal embedding dimension of the causal variable
    libraryMax : int
        Maximum library size
    theta : int
        Theta value to use for EDM calls.
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether to print the progress bar for EDM calls.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    causalSummary : str
        Strength of evidence for causation. Either 'Strong evidence', 'Some evidence', 
        or 'No evidence'.
    ------------------------------------------------------------------------------------------
    '''
    
    # Make some rough guesses for the Monster exponential fit coefficients
    ccm = cross_mapping(t, x, y, E_best, libraryMax, theta, verbosity, showProgressBar)
    ccmRes = ccm["summary"]
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

    if verbosity > 1:
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

    return causalSummary

def test_convergence_dist(t, x, y, E_best, libraryMax, theta, verbosity, showProgressBar, numReps = 1000):
    '''
    Test for convergence by comparing the distribution of rho at a small library size and
    a sampled rho at the maximum library size.
    ------------------------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------------------------
    t : np.ndarray
        The sampling times for cause/effect time series.
    x : np.ndarray
        The causal time series.
    y : np.ndarray
        The effect time series.
    E_best : int
        Optimal embedding dimension of the causal variable
    libraryMax : int
        Maximum library size
    theta : int
        Theta value to use for EDM calls.
    verbosity : int
        The level of detail in the output. 0 for final result only, 1 for results of each step,
        2 for subheading, 3 for additional plots
    showProgressBar : bool
        Whether to print the progress bar for EDM calls.
    ------------------------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------------------------
    causalSummary : str
        Strength of evidence for causation. Either 'Strong evidence', 'Some evidence', 
        or 'No evidence'.
    ------------------------------------------------------------------------------------------
    '''
    librarySmall = max(E_best + 2, libraryMax // 10)

    distRes = edm(t, y, x, E = E_best, library = librarySmall, numReps = numReps, theta = theta,
                  algorithm = "simplex", k = np.inf, shuffle = True, verbosity = 0, 
                  showProgressBar = showProgressBar)
    dist = distRes['stats']['rho']
    
    finalRes = edm(t, y, x, E = E_best, library = libraryMax, theta = theta,  
                algorithm = "simplex", k = np.inf, shuffle = True, verbosity = 0, 
                showProgressBar = showProgressBar)
    finalRho = float(finalRes["summary"]["rho"])
    
    q975, q95 = np.quantile(dist, 0.975), np.quantile(dist, 0.95)
    rhoQuantile = np.count_nonzero(dist<finalRho) / dist.size
    if verbosity >= 1:
        print(f"At library=E+2, found rho quantiles of {round(q975, 5)} (0.975) and {round(q95, 5)} (0.95)")
        print(f"At library=max, found final rho was {round(finalRho, 5)}, i.e. quantile={rhoQuantile}")
    
    if (finalRho > q975):
        causalSummary = "Strong evidence"
    elif (finalRho > q95):
        causalSummary = "Some evidence"
    else:
        causalSummary = "No evidence"    
        
    return causalSummary

