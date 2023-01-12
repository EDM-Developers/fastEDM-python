import numpy as np


def tslag(t, x, lag=1, dt=1):
    t = np.asarray(t)
    x = np.asarray(x)
    l_x = np.full(len(t), np.nan)
    for i in range(len(t)):
        lagged_t = t[i] - lag * dt
        if not np.isnan(lagged_t) and lagged_t in t:
            l_x[i] = x[t == lagged_t]
    return l_x


def tsdiff(t, x, lag=1, dt=1):
    t = np.asarray(t)
    x = np.asarray(x)
    d_x = np.full(len(t), np.nan)
    for i in range(len(t)):
        lagged_t = t[i] - lag * dt
        if not np.isnan(x[i]) and not np.isnan(lagged_t) and lagged_t in t:
            d_x[i] = x[i] - x[t == lagged_t]
    return d_x


def logistic_map(obs):
    r_x = 3.625
    r_y = 3.77
    beta_xy = 0.05
    beta_yx = 0.4
    tau = 1

    x = np.full(obs, np.nan)
    y = np.full(obs, np.nan)

    x[0] = 0.2
    y[0] = 0.4

    for i in range(1, obs):
        x[i] = x[i - 1] * (r_x * (1 - x[i - 1]) - beta_xy * y[i - 1])
        y[i] = y[i - 1] * (r_y * (1 - y[i - 1]) - beta_yx * x[i - tau])

    return (x, y)


def expect_approx_equal(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    validInputs = (
        len(x) == len(y)
        and len(x) == np.sum(np.isfinite(x))
        and len(y) == np.sum(np.isfinite(y))
    )

    assert validInputs

    if validInputs:
        absErr = np.max(np.abs(x - y))
        assert absErr < 1e-4


def check_edm_result(res, rho, co_rho=None):
    assert res["rc"] == 0
    expect_approx_equal(res["stats"]["rho"], rho)

    if co_rho is not None:
        expect_approx_equal(res["copredStats"]["rho"], co_rho)


def check_edm_results(res1, res2, rho1, rho2):
    check_edm_result(res1, rho1)
    check_edm_result(res2, rho2)


def check_noisy_edm_result(res, rho_1, rho_2, co_rho_1=None, co_rho_2=None):
    assert res["rc"] == 0

    df = res["stats"].dropna()
    meanRho = df.groupby(["E", "library", "theta"])[["rho"]].mean()
    meanRho = np.asarray(meanRho).flatten()
    assert np.all((np.asarray(rho_1) <= meanRho) & (meanRho <= np.asarray(rho_2)))

    if co_rho_1 is not None:
        df = res["copredStats"].dropna()
        meanCoRho = df.groupby(["E", "library", "theta"])[["rho"]].mean()
        meanCoRho = np.asarray(meanCoRho).flatten()
        assert np.all(
            (np.asarray(co_rho_1) <= meanCoRho) & (meanCoRho <= np.asarray(co_rho_2))
        )

def expand(matrix, predictionRows):
    expanded = []
    obsNum = 0
    for use in predictionRows:
        if use:
            expanded.append(matrix[obsNum])
            obsNum += 1
        else:
            expanded.append(np.full(matrix.shape[1:], np.NAN))
    
    return np.array(expanded)

# Some uniform rv's from Stata using the seed '12345678'
UNIF_200 = [
    0.4032161652730771,
    0.9814345444173197,
    0.3373750503685581,
    0.1791857833127429,
    0.2497615187767713,
    0.2969102692610567,
    0.2049932720581052,
    0.6127560679465497,
    0.1945775905414523,
    0.2190264344998417,
    0.5967941090343319,
    0.2732545386445274,
    0.783274501879668,
    0.5998517662013346,
    0.839228334965931,
    0.9191838391294088,
    0.2934124279570279,
    0.6891432790291355,
    0.2048246627394067,
    0.4428944844872612,
    0.5035322526301146,
    0.7846905293507423,
    0.3973196413139125,
    0.1417179025397167,
    0.2297586957409598,
    0.4222709916638957,
    0.8461065233021708,
    0.0807904559242613,
    0.1544088321271263,
    0.5613257424648498,
    0.0784469845635632,
    0.7530032474630725,
    0.0137490402736743,
    0.1284542865245989,
    0.8463379622597907,
    0.9969955966978709,
    0.1686659431226015,
    0.1173465910722692,
    0.5364226240241997,
    0.5159901701260331,
    0.6727813123885479,
    0.9625629665008296,
    0.9595417075442659,
    0.1172940837748137,
    0.5654193787312168,
    0.8433041905223945,
    0.1371089038434833,
    0.3894521245853525,
    0.3767744584921785,
    0.5154827677849309,
    0.5144958419236422,
    0.5392356674361488,
    0.0602919934618357,
    0.5842425522598768,
    0.1127997590422875,
    0.3017396141777563,
    0.566056123747399,
    0.0234300500483061,
    0.446666137907659,
    0.8325948629304355,
    0.3859420229362572,
    0.5913344449911008,
    0.5429686057208226,
    0.8475534057216773,
    0.2548978318715515,
    0.5401877915192624,
    0.6490722076532688,
    0.9978749125996588,
    0.8603831590200534,
    0.0646047336240946,
    0.3366168869966312,
    0.0529183026796373,
    0.2775160253539241,
    0.13714739352711,
    0.5332886602574597,
    0.4554481474468138,
    0.2554947058562992,
    0.2117942810144512,
    0.5143683948513994,
    0.4109627526795842,
    0.1152777732355069,
    0.7413683768433018,
    0.3061423425767477,
    0.7863824045156084,
    0.6039497263555896,
    0.0291932555292082,
    0.5231675914641521,
    0.1969319200248217,
    0.5935410685602686,
    0.2641492447722841,
    0.2907241830474051,
    0.3784841046138104,
    0.9331586394050942,
    0.9605276395327141,
    0.5655670284878619,
    0.5222821795008065,
    0.8516779163207383,
    0.1100451566808257,
    0.7957954273960438,
    0.7962176143105376,
    0.2354088022022441,
    0.9382453662561788,
    0.8192266352790453,
    0.4144106771809849,
    0.490977891655796,
    0.9127480800163904,
    0.0657145839743215,
    0.4519554710752521,
    0.0709059022401829,
    0.2662836024528649,
    0.5345772940845596,
    0.4265899010392383,
    0.2747081587127954,
    0.5164019507468386,
    0.8349998398567205,
    0.107226227911105,
    0.4905188464764981,
    0.7242481387577447,
    0.9682545409150528,
    0.9822068122956905,
    0.4793603880336778,
    0.8485766361764732,
    0.9406273141585527,
    0.6095116145179326,
    0.6085852384563054,
    0.4529303236044436,
    0.2757519773710629,
    0.0762361381946745,
    0.4452929061484255,
    0.0908948151163734,
    0.5022906216805424,
    0.8089346066118283,
    0.9158658613180911,
    0.9969150573511568,
    0.3558791562632609,
    0.8614877244157376,
    0.3765989177730652,
    0.2453749846785717,
    0.6009273882813092,
    0.8352408531141619,
    0.6661539258782491,
    0.7710569642593318,
    0.7624201378973611,
    0.9931531663177428,
    0.923034959330067,
    0.7242071002415561,
    0.7560926864138929,
    0.093896026984115,
    0.4879173119413863,
    0.913485895383184,
    0.1299684863347993,
    0.1036428325657651,
    0.6880189676686744,
    0.5026425620798053,
    0.648417138502822,
    0.8448840509985992,
    0.4874908670167216,
    0.5784980295332124,
    0.8381816958913403,
    0.4027480031037615,
    0.6671161681380354,
    0.3396350096610617,
    0.0232787820504924,
    0.265128654258174,
    0.4766564574754034,
    0.0782706739728657,
    0.6127005528438244,
    0.4072321046516864,
    0.5227203330605766,
    0.1977372747415979,
    0.7507934791611285,
    0.0978892111143869,
    0.1836887170888787,
    0.6029686128691382,
    0.5197504708621133,
    0.2416433912954606,
    0.6078390094509868,
    0.9966368006778291,
    0.8066905771208216,
    0.2479132038263475,
    0.1210908289739182,
    0.5033590914183873,
    0.5794444917633742,
    0.3891651009205163,
    0.5349964783397266,
    0.1912104418458377,
    0.0509169682297838,
    0.8096895160137013,
    0.4719385020310226,
    0.6040194781564009,
    0.9915954480975472,
    0.0625292627821351,
    0.0013004956320272,
    0.8333968865756917,
    0.1333932163423718,
    0.3031928113136991,
    0.5646750635559817,
    0.3671586644732209,
    0.4074915152964259,
    0.8944071321855526,
    0.3981792341304314,
]

# Some uniform rv's from Stata using the seed '12345678'
UNIF_100 = [
    0.4032161652730771,
    0.9814345444173197,
    0.3373750503685581,
    0.1791857833127429,
    0.2497615187767713,
    0.2969102692610567,
    0.2049932720581052,
    0.6127560679465497,
    0.1945775905414523,
    0.2190264344998417,
    0.5967941090343319,
    0.2732545386445274,
    0.783274501879668,
    0.5998517662013346,
    0.839228334965931,
    0.9191838391294088,
    0.2934124279570279,
    0.6891432790291355,
    0.2048246627394067,
    0.4428944844872612,
    0.5035322526301146,
    0.7846905293507423,
    0.3973196413139125,
    0.1417179025397167,
    0.2297586957409598,
    0.4222709916638957,
    0.8461065233021708,
    0.0807904559242613,
    0.1544088321271263,
    0.5613257424648498,
    0.0784469845635632,
    0.7530032474630725,
    0.0137490402736743,
    0.1284542865245989,
    0.8463379622597907,
    0.9969955966978709,
    0.1686659431226015,
    0.1173465910722692,
    0.5364226240241997,
    0.5159901701260331,
    0.6727813123885479,
    0.9625629665008296,
    0.9595417075442659,
    0.1172940837748137,
    0.5654193787312168,
    0.8433041905223945,
    0.1371089038434833,
    0.3894521245853525,
    0.3767744584921785,
    0.5154827677849309,
    0.5144958419236422,
    0.5392356674361488,
    0.0602919934618357,
    0.5842425522598768,
    0.1127997590422875,
    0.3017396141777563,
    0.566056123747399,
    0.0234300500483061,
    0.446666137907659,
    0.8325948629304355,
    0.3859420229362572,
    0.5913344449911008,
    0.5429686057208226,
    0.8475534057216773,
    0.2548978318715515,
    0.5401877915192624,
    0.6490722076532688,
    0.9978749125996588,
    0.8603831590200534,
    0.0646047336240946,
    0.3366168869966312,
    0.0529183026796373,
    0.2775160253539241,
    0.13714739352711,
    0.5332886602574597,
    0.4554481474468138,
    0.2554947058562992,
    0.2117942810144512,
    0.5143683948513994,
    0.4109627526795842,
    0.1152777732355069,
    0.7413683768433018,
    0.3061423425767477,
    0.7863824045156084,
    0.6039497263555896,
    0.0291932555292082,
    0.5231675914641521,
    0.1969319200248217,
    0.5935410685602686,
    0.2641492447722841,
    0.2907241830474051,
    0.3784841046138104,
    0.9331586394050942,
    0.9605276395327141,
    0.5655670284878619,
    0.5222821795008065,
    0.8516779163207383,
    0.1100451566808257,
    0.7957954273960438,
    0.7962176143105376,
]

# Some normal rv's from Stata using the seed '1'
NORM_100 = [
    0.94173813,
    0.4870331,
    0.55453211,
    -0.57394189,
    -1.6831859,
    0.20002605,
    2.0535631,
    -1.2874906,
    0.76769561,
    0.57129043,
    -0.9382565,
    1.4670297,
    -2.7969353,
    0.65672988,
    -0.074978352,
    -0.61362195,
    -1.3412304,
    0.45943514,
    1.1464604,
    1.3768886,
    0.016770668,
    0.94677925,
    -0.11319048,
    -0.49819016,
    -1.5304253,
    -0.051611003,
    -0.076513439,
    -1.3290932,
    -0.45883241,
    0.017877782,
    0.34325397,
    1.2092726,
    0.2365011,
    -0.73019648,
    -0.330953,
    0.13359453,
    1.0885595,
    -0.63763547,
    -0.42640716,
    -0.014303211,
    0.21588294,
    0.05830165,
    0.059484873,
    0.025059106,
    1.0119363,
    -0.35853708,
    1.4637038,
    0.70681834,
    -2.8081942,
    -0.27054599,
    1.5580958,
    0.071366407,
    2.2807562,
    0.92863506,
    -0.16536251,
    -0.17245923,
    2.0830457,
    -1.6134628,
    -0.16830915,
    1.6171873,
    -0.90855205,
    0.0026675737,
    0.82025963,
    0.92624164,
    1.6329502,
    -0.232575,
    -0.089815319,
    -1.0917373,
    0.061252236,
    1.1413523,
    -0.0335248,
    0.26932761,
    -1.9740542,
    -0.99436063,
    -0.53038871,
    0.70026708,
    -0.79605526,
    -1.1729968,
    0.17358617,
    -0.28859794,
    0.93706262,
    1.2917892,
    -0.06885922,
    1.0749949,
    1.3219627,
    -0.093162067,
    1.0999831,
    0.31230453,
    -0.87349302,
    1.4867147,
    -0.8970021,
    -1.1020641,
    0.25990388,
    -1.9723424,
    1.5126398,
    1.4318892,
    -0.024286436,
    -0.33137387,
    -0.64844704,
    -1.7218629,
]

NORM_200 = [
    -1.027216,
    1.527376,
    0.5907618,
    1.070512,
    2.139774,
    -0.5876155,
    0.1418234,
    1.390853,
    -1.030574,
    0.5835255,
    1.538284,
    1.095415,
    1.289363,
    0.4250214,
    1.332112,
    0.1224301,
    0.4007208,
    1.163034,
    -0.9338163,
    -1.553558,
    1.128875,
    0.71824,
    0.8828724,
    -0.9635994,
    0.5716761,
    0.0727569,
    -0.3750865,
    -0.8911737,
    -0.8376914,
    -0.3425734,
    -1.895796,
    1.220617,
    0.8647164,
    -0.4872026,
    0.1291741,
    -1.807868,
    0.9658784,
    -0.8437532,
    0.7287974,
    -0.0579607,
    -0.7721093,
    0.3223931,
    0.4673252,
    -0.3628134,
    -0.8418728,
    -0.8550454,
    -1.341583,
    -0.4182656,
    0.4155265,
    -0.3210205,
    0.7979518,
    0.0385472,
    -2.345896,
    -0.0535184,
    -1.997315,
    -0.897661,
    -1.172937,
    -1.374793,
    -0.439018,
    1.212688,
    -0.8391462,
    -0.2125729,
    0.3922674,
    -1.24292,
    -0.3563064,
    -1.368325,
    1.293824,
    -1.078043,
    -0.6217906,
    0.2247944,
    -0.3572458,
    1.455859,
    0.177133,
    -0.4954876,
    -0.4623527,
    -0.9394832,
    -1.381252,
    0.3134706,
    0.1598284,
    0.4492666,
    0.7745574,
    2.02939,
    0.2769991,
    -1.729418,
    -0.0719662,
    -0.4887659,
    -0.6402079,
    -0.3815501,
    -0.6201261,
    -0.6295606,
    0.2707956,
    1.056473,
    -1.657482,
    1.228817,
    0.8577658,
    0.4940666,
    1.37631,
    -0.0235891,
    1.044822,
    0.2835678,
    0.019814,
    -1.331117,
    -0.4936376,
    -1.570097,
    1.482886,
    -0.2730185,
    -0.467406,
    0.8039773,
    0.6066654,
    0.099022,
    1.246193,
    -0.6019896,
    -1.078758,
    0.0527143,
    0.522496,
    0.7971591,
    2.091462,
    -1.87791,
    1.123751,
    0.1762845,
    1.552169,
    -0.4524258,
    0.4963196,
    -1.343762,
    1.630493,
    -0.1519897,
    0.4249264,
    0.1730838,
    -1.662154,
    0.5415513,
    1.762257,
    0.4248972,
    -1.56878,
    -0.0073573,
    0.4523424,
    -1.077807,
    -3.545176,
    -1.198717,
    1.314406,
    -1.067673,
    -0.7234299,
    1.150322,
    2.114344,
    0.4767627,
    1.228333,
    1.247601,
    -0.2687568,
    1.233031,
    1.063017,
    -1.619441,
    0.5857949,
    1.296269,
    0.8043274,
    0.3258621,
    3.569143,
    0.3741727,
    -1.49533,
    -0.0184031,
    0.2356096,
    -1.738142,
    -0.3104737,
    -0.377933,
    -0.5639113,
    -1.457661,
    0.9921553,
    -0.9124324,
    -0.0439041,
    -0.6419182,
    0.5668358,
    -0.4034521,
    -0.3590932,
    -1.489591,
    -0.5190973,
    0.5887823,
    0.8400694,
    0.0363247,
    1.122107,
    -0.0369949,
    1.10605,
    0.6818572,
    -0.1490808,
    -0.9733297,
    -0.8749319,
    0.6384861,
    -1.647552,
    -2.270525,
    0.6330903,
    0.1588243,
    -0.0146699,
    -0.2460195,
    0.7494598,
    -0.0442753,
    -1.198142,
    -0.1973266,
    0.7962075,
    -0.0928933,
    2.165736,
    -0.7527414,
    1.006963,
    0.1770673,
    -0.4803994,
]
