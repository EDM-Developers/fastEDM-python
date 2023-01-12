import numpy as np
import pandas as pd
import unittest

from fastEDM import edm, create_manifold
from helper import *


class TestCI(unittest.TestCase):
    def test_logistic_map(self):
        x, y = logistic_map(4)
        assert np.allclose(x, [0.2000000, 0.5760000, 0.8601754, 0.4266398])
        assert np.allclose(y, [0.4000000, 0.8728000, 0.2174529, 0.5667110])

    def test_create_manifold(self):
        # Basic manifold creation [basicManifold]
        E = 2
        tau = 1
        p = 1
        t = [1, 2, 3, 4]
        x = [11, 12, 13, 14]

        # Basic manifold, no extras or dt
        manifold = create_manifold(t, x, E=E, tau=tau, p=p)
        true_manifold = np.array([[12, 11], [13, 12]])
        assert np.allclose(manifold, true_manifold)

    # formals(edm)$verbosity = 0
    # formals(edm)$showProgressBar = False

    def test_no_seed_predict_past(self):
        "No seed and predictWithPast=True"
        obs = 10
        t = np.arange(1, obs + 1)
        x = np.arange(1, obs + 1)
        edm(t, x, shuffle=True)

        # Make sure the plugin doesn't crash if 'predictWithPast' is set
        old = edm(t, x, full=True, predictWithPast=False, savePredictions=True)
        new = edm(t, x, full=True, predictWithPast=True, savePredictions=True)

        check_edm_result(old, 0.9722)
        check_edm_result(new, 0.99866)

    def test_simple_manifold(self):
        "Simple manifolds"
        obs = 500
        x, y = logistic_map(obs)

        t = np.arange(1, len(x) + 1, dtype=float)
        x = x[299:]
        y = y[299:]
        t = t[299:]

        # Some normal rv's from Stata using the seed '12345678'
        u1 = np.array(NORM_200)

        # explore x, e(2/10)
        res = edm(t, x, E=range(2, 10 + 1))
        rho = np.array(
            [
                0.99893,
                0.99879,
                0.99835,
                0.99763,
                0.99457,
                0.99385,
                0.991,
                0.98972,
                0.98572,
            ]
        )
        check_edm_result(res, rho)

        # edm xmap x y, k(5)
        res1 = edm(t, x, y, k=5)
        res2 = edm(t, y, x, k=5)
        check_edm_results(res1, res2, 0.55861, 0.94454)

        # edm xmap x y, e(6) lib(8)
        res1 = edm(t, x, y, E=6, library=8)
        res2 = edm(t, y, x, E=6, library=8)
        check_edm_results(res1, res2, 0.3362, 0.51116)

        # edm explore x, k(5) crossfold(10)
        res = edm(t, x, k=5, crossfold=10)
        expect_approx_equal(np.mean(res["stats"]["rho"]), 0.99946)

        # edm explore x, theta(0.2(0.1)2.0) algorithm(smap)
        res = edm(t, x, theta=np.arange(0.2, 2.0 + 0.1, 0.1), algorithm="smap")
        expect_approx_equal(res["stats"]["rho"].iloc[0], 0.99874)
        expect_approx_equal(res["stats"]["rho"].iloc[-1], 0.99882)

        # edm xmap x y, theta(0.2) algorithm(smap) savesmap(beta)
        res1 = edm(t, x, y, theta=0.2, algorithm="smap", saveSMAPCoeffs=True)
        res2 = edm(t, y, x, theta=0.2, algorithm="smap", saveSMAPCoeffs=True)
        beta1 = res1["coeffs"]
        check_edm_results(res1, res2, 0.66867, 0.98487)

        # No missing values in coefficients
        assert np.sum(np.isnan(beta1)) == 0
    
        # assert beta1_b2_rep1 != . if _n > 1
        beta1 = expand(beta1, res1["predictionRows"])
        assert np.sum(np.isnan(beta1[0, :])) == beta1.shape[1]
        assert np.sum(np.isnan(beta1[1:])) == 0

        # edm xmap y x, predict(x2) direction(oneway)
        res = edm(t, y, x, savePredictions=True)
        x2 = res["predictions"]
        check_edm_result(res, 0.94272)

        # No missing values in predictions
        assert sum(np.isnan(x2)) == 0

        # assert x2 != . if _n > 1
        x2 = expand(x2, res["predictionRows"])
        assert np.isnan(x2[0])
        assert sum(np.isnan(x2[1:])) == 0

        # edm explore x, copredict(teste) copredictvar(y)
        res = edm(t, x, copredict=y, saveCoPredictions=True)
        teste = res["copredictions"]
        check_edm_result(res, 0.9989, co_rho=0.78002)

        # No missing values in copredictions
        assert np.sum(np.isnan(teste)) == 0

        # assert teste != . if _n > 1
        teste = expand(teste, res["predictionRows"])
        assert np.isnan(teste[0])
        assert np.sum(np.isnan(teste[1:])) == 0

        # edm explore z.x, p(10)
        z_x = (x - np.nanmean(x)) / np.nanstd(
            x
        )  # This is slightly different to Stata ('touse' perhaps)
        res = edm(t, z_x, p=10)
        check_edm_result(res, 0.90235)

        # edm xmap y x, p(10) direction(oneway)
        res = edm(t, y, x, p=10)
        check_edm_result(res, 0.89554)

        # edm xmap y x, p(10) copredict(testx) copredictvar(x2) direction(oneway)
        res = edm(t, y, x, p=10, copredict=x2, saveCoPredictions=True)
        testx = res["copredictions"]
        check_edm_result(res, 0.89554, co_rho=0.67401)

        # No missing values in copredictions.
        assert np.sum(np.isnan(testx)) == 0

        # assert testx != . if _n >= 3
        testx = expand(testx, res["predictionRows"])
        assert np.sum(np.isnan(testx[:2])) == 2
        assert np.sum(np.isnan(testx[2:])) == 0

        # edm xmap y x, p(10) copredict(testx2) copredictvar(z.x2) direction(oneway)
        # In Python, we would do:
        #   z_x2 = (x2 - np.nanmean(x2)) / np.nanstd(x2)
        # However, the np.nanmean/np.nanstd have more precision than the R equivalent.
        # So to match R's results, here are the lower-precision versions of those quantities.
        z_x2 = (x2 - 0.6404819) / 0.216451
        res = edm(t, y, x, p=10, copredict=z_x2, saveCoPredictions=True)
        testx2 = res["copredictions"]
        check_edm_result(res, 0.89554, co_rho=0.93837)

        # No missing values in copredictions
        assert np.sum(np.isnan(testx2)) == 0

        # assert testx2 != . if _n >= 3
        testx2 = expand(testx2, res["predictionRows"])
        assert np.sum(np.isnan(testx2[:2])) == 2
        assert np.sum(np.isnan(testx2[2:])) == 0

        # edm xmap y x, extra(u1) p(10) copredict(testx3) copredictvar(z.x2) direction(oneway)
        res = edm(t, y, x, extras=[u1], p=10, copredict=z_x2, saveCoPredictions=True)
        testx3 = res["copredictions"]
        check_edm_result(res, 0.37011, co_rho=0.9364)

        # No missing values in copredictions.
        assert np.sum(np.isnan(testx3)) == 0

        # assert testx3 != . if _n >= 3
        testx3 = expand(testx3, res["predictionRows"])
        assert np.sum(np.isnan(testx3[:2])) == 2
        assert np.sum(np.isnan(testx3[2:])) == 0

        # Check explore / xmap consistency

        # edm xmap l.x x, direction(oneway)
        resXmap = edm(t, tslag(t, x), x)
        check_edm_result(resXmap, 0.99939)

        # edm explore x, full
        resExplore = edm(t, x, full=True)
        check_edm_result(resExplore, 0.99939)

        # assert xmap_r[1,1] == explore_r[1,1]
        expect_approx_equal(resXmap["stats"]["rho"], resExplore["stats"]["rho"])

        # Check xmap reverse consistency (not necessary to check in this version)
        res1 = edm(t, x, y)
        res2 = edm(t, y, x)
        check_edm_results(res1, res2, 0.54213, 0.94272)

        # Make sure multiple e's and multiple theta's work together

        # edm explore x, e(2 3) theta(0 1)
        res = edm(t, x, E=[2, 3], theta=[0, 1])
        rho = [0.99863, 0.99895, 0.99734, 0.99872]
        check_edm_result(res, rho)

        # Check that lowmemory flag is working
        res = edm(t, x, lowMemory=True)
        check_edm_result(res, 0.9989)

        # Check that verbosity > 0 is working
        res = edm(t, x, verbosity=1)  # TODO: Find Python version of R's capture.output
        check_edm_result(res, 0.9989)

        # Check that numThreads > 1 is working
        res = edm(t, x, numThreads=4)
        check_edm_result(res, 0.9989)

    def test_missing_data_manifolds(self):
        obs = 500
        x, y = logistic_map(obs)

        t = np.arange(1, len(x) + 1, dtype=float)
        x = x[299:]
        y = y[299:]
        t = t[299:]

        # Some uniform rv's from Stata using the seed '12345678'
        u = UNIF_200

        # Test missing data
        df = pd.DataFrame({"t": t, "x": x, "y": y, "u": u})

        df = df.loc[df.u >= 0.1]
        df.loc[df.u < 0.2, "x"] = np.nan
        df.loc[df.t % 19 == 1, "t"] = np.nan

        t = df.t
        x = df.x
        y = df.y
        u = df.u

        # edm explore x
        res = edm(t, x)
        check_edm_result(res, 0.99814)

        # edm explore x, dt savemanifold(plugin) dtweight(1)
        res = edm(t, x, dt=True, saveManifolds=True, dtWeight=1)
        check_edm_result(res, 0.95569)

        # edm explore x, allowmissing
        res = edm(t, x, allowMissing=True)
        check_edm_result(res, 0.99766)

        # edm explore x, missingdistance(1)
        res = edm(t, x, allowMissing=True, missingDistance=1.0)
        check_edm_result(res, 0.99765)

        # TODO: Decide whether this is better -- being explicit about 'allowMissing' & 'missingDistance'
        # or whether to follow Stata and just let the latter auto-enable the former...

        # edm xmap x l.x, allowmissing
        res1 = edm(t, x, tslag(t, x), allowMissing=True)
        res2 = edm(t, tslag(t, x), x, allowMissing=True)
        check_edm_results(res1, res2, 0.99983, 0.99864)

        # edm xmap x l.x, extraembed(u) dt alg(smap) savesmap(newb) e(5)
        res1 = edm(
            t,
            x,
            tslag(t, x),
            extras=[u],
            dt=True,
            algorithm="smap",
            saveSMAPCoeffs=True,
            E=5,
        )
        res2 = edm(
            t,
            tslag(t, x),
            x,
            extras=[u],
            dt=True,
            algorithm="smap",
            saveSMAPCoeffs=True,
            E=5,
        )
        check_edm_results(res1, res2, 1.0, 0.77523)

        # edm xmap x l3.x, extraembed(u) dt alg(smap) savesmap(newc) e(5) oneway dtsave(testdt)
        res = edm(
            t,
            x,
            tslag(t, x, 3),
            extras=[u],
            dt=True,
            algorithm="smap",
            saveSMAPCoeffs=True,
            E=5,
        )
        check_edm_result(res, 0.36976)

        # edm explore x, extraembed(u) allowmissing dt crossfold(5)
        res = edm(t, x, extras=[u], allowMissing=True, dt=True, crossfold=5)
        expect_approx_equal(np.mean(res["stats"]["rho"]), 0.92512)

        # edm explore d.x, dt
        res = edm(t, tsdiff(t, x), dt=True)
        check_edm_result(res, 0.89192)

        # edm explore x, rep(20) ci(95)
        res = edm(t, x, numReps=20)
        check_noisy_edm_result(res, 0.99225, 0.9981)

        # edm xmap x y, lib(50) rep(20) ci(95)
        res1 = edm(t, x, y, library=50, numReps=20)
        res2 = edm(t, y, x, library=50, numReps=20)
        check_noisy_edm_result(res1, 0.35556, 0.40613)
        check_noisy_edm_result(res2, 0.82245, 0.85151)

    def test_bigger_script(self):
        obs = 100
        x, y = logistic_map(obs)
        t = np.arange(1, len(x) + 1)

        # Some uniform rv's from Stata using the seed '12345678'
        u = UNIF_100

        # Some normal rv's from Stata using the seed '1'
        u1 = NORM_100

        # edm explore x, e(2) crossfold(2) k(-1) allowmissing
        res = edm(t, x, E=2, crossfold=2, k=float("inf"), allowMissing=True)
        expect_approx_equal(np.mean(res["stats"]["rho"]), 0.98175)
        # TODO: Make the crossfold option just output one correlation

        # edm explore x, e(2) crossfold(10) k(-1) allowmissing
        res = edm(t, x, E=2, crossfold=10, k=float("inf"), allowMissing=True)
        expect_approx_equal(np.mean(res["stats"]["rho"]), 0.98325)

        # edm explore x, e(5) extra(d.y) full allowmissing
        res = edm(t, x, E=5, extras=[tsdiff(t, y)], full=True, allowMissing=True)
        check_edm_result(res, 0.95266)

        # Introduce missing data and test all the dt variations
        df = pd.DataFrame({"t": t, "x": x, "y": y, "u": u, "u1": u1})
        df = df.loc[df.u >= 0.1]
        df.loc[df.u < 0.2, "x"] = np.nan
        df.loc[df.t % 7 == 1, "u1"] = np.nan
        df.loc[df.t % 19 == 1, "t"] = np.nan

        t = df.t
        x = df.x
        y = df.y
        u = df.u
        u1 = df.u1

        # Make sure multiple library values are respected

        # edm xmap x y, allowmissing dt library(10(5)70)
        res1 = edm(t, x, y, allowMissing=True, dt=True, library=range(10, 70 + 5, 5))
        res2 = edm(t, y, x, allowMissing=True, dt=True, library=range(10, 70 + 5, 5))

        rho1 = [
            0.20492,
            0.11316,
            0.15244,
            0.18469,
            0.2577,
            0.28964,
            0.29208,
            0.33099,
            0.39233,
            0.41628,
            0.37522,
            0.36816,
            0.40495,
        ]
        rho2 = [
            0.39118,
            0.55506,
            0.6788,
            0.70348,
            0.71176,
            0.72476,
            0.75539,
            0.78565,
            0.80807,
            0.83358,
            0.83503,
            0.85401,
            0.85847,
        ]
        check_edm_results(res1, res2, rho1, rho2)

        # See if the negative values of p are allowed

        # edm explore x, p(-1)
        res = edm(t, x, p=-1)
        check_edm_result(res, 0.99751)

        # edm xmap x y, p(-1)
        res1 = edm(t, x, y, p=-1)
        res2 = edm(t, y, x, p=-1)
        check_edm_results(res1, res2, 0.26842, 0.8974)

        # Try out copredict and copredictvar combinations with multiple reps etc.

        # edm explore x, copredictvar(y)
        res = edm(t, x, copredict=y)
        check_edm_result(res, 0.99237, co_rho=0.67756)

        # edm explore x, copredictvar(y) full
        res = edm(t, x, copredict=y, full=True)
        check_edm_result(res, 0.99416, co_rho=0.77599)

        # edm xmap x y, copredictvar(u1)
        res1 = edm(t, x, y, copredict=u1)
        res2 = edm(t, y, x, copredict=u1)
        check_edm_result(res1, 0.30789, co_rho=0.42901)
        check_edm_result(res2, 0.90401, co_rho=0.5207)

        # Note the E=5, theta=0 predictions are all the exact same value
        # so the correlation being '.' is actually correct.

        # edm explore x, e(2/5) theta(0 1) copredictvar(y)
        res = edm(t, x, E=range(2, 5 + 1), theta=[0, 1], copredict=y)

        rho = [0.90482, 0.95631, 0.88553, 0.95751, 0.90482, 0.95652, None, 0.95565]
        co_rho = [0.47353, 0.51137, 0.41523, 0.50186, 0.27504, 0.42485, None, 0.48008]

        assert res["rc"] == 0

        resRhoDropped = res["stats"]["rho"].drop(7)
        rhoDropped = np.delete(rho, 7)
        absErr = np.max(np.abs(resRhoDropped - rhoDropped))
        assert absErr < 1e-4

        resRhoDropped = res["copredStats"]["rho"].drop(7)
        rhoDropped = np.delete(co_rho, 7)
        absErr = np.max(np.abs(resRhoDropped - rhoDropped))
        assert absErr < 1e-4

        # edm xmap x y, library(5 10 20 40) copredictvar(u1)
        res1 = edm(t, x, y, library=[5, 10, 20, 40], copredict=u1)
        res2 = edm(t, y, x, library=[5, 10, 20, 40], copredict=u1)

        rho = [0.18385, 0.085223, 0.085659, 0.22313]
        co_rho = [0.18429, 0.26729, 0.37307, 0.36359]
        check_edm_result(res1, rho, co_rho)

        rho = [0.43651, 0.49275, 0.71521, 0.84646]
        co_rho = [0.63167, 0.63089, 0.50528, 0.3571]
        check_edm_result(res2, rho, co_rho)

        # edm explore x, copredictvar(y) rep(20)
        res = edm(t, x, copredict=y, numReps=20)
        check_noisy_edm_result(res, 0.97335, 0.99339, 0.67584, 0.71214)

        # edm xmap x y, library(5 10 20 40) copredictvar(u1) rep(100)
        res1 = edm(t, x, y, library=[5, 10, 20, 40], copredict=u1, numReps=100)
        res2 = edm(t, y, x, library=[5, 10, 20, 40], copredict=u1, numReps=100)

        rho_low = [-0.24081, -0.040511, 0.055321, 0.22091]
        rho_up = [0.74934, 0.63791, 0.53273, 0.4299]
        co_rho_low = [-0.40689, 0.1645, 0.15547, 0.045263]
        co_rho_up = [1.1666, 0.93126, 0.86293, 0.70085]

        check_noisy_edm_result(res1, rho_low, rho_up, co_rho_low, co_rho_up)

        rho_low = [-0.15045, 0.25476, 0.53274, 0.72057]
        rho_up = [0.83649, 0.78668, 0.83097, 0.90064]
        co_rho_low = [-0.38777, 0.062654, 0.32211, 0.39681]
        co_rho_up = [1.0946, 0.97872, 0.7964, 0.67552]

        check_noisy_edm_result(res2, rho_low, rho_up, co_rho_low, co_rho_up)

        # # edm explore x, copredictvar(y) rep(100) ci(10)
        # res = edm(t, x, copredict=y, numReps=100)
        #
        # # edm xmap x y, library(5 10 20 40) copredictvar(u1) rep(4) detail
        # res1 = edm(t, x, y, library=[5, 10, 20, 40], copredict=u1, numReps=4)
        # res2 = edm(t, y, x, library=[5, 10, 20, 40], copredict=u1, numReps=4)

    def test_panel_data(self):
        obs = 100
        x, y = logistic_map(obs)
        t = np.arange(1, len(x) + 1)
        panel = (t > obs / 3).astype(int)

        # edm explore x, e(40)
        res = edm(t, x, panel=panel, E=40)
        check_edm_result(res, 0.86964)

        # edm explore x, e(40) allowmissing
        res = edm(t, x, panel=panel, E=40, allowMissing=True)
        check_edm_result(res, 0.92115)

        # edm explore x, e(40) idw(-1)
        res = edm(t, x, panel=panel, E=40, panelWeight=float("inf"))
        check_edm_result(res, 0.86964)

        # edm explore x, e(40) idw(-1) allowmissing
        res = edm(t, x, panel=panel, E=40, panelWeight=float("inf"), allowMissing=True)
        check_edm_result(res, 0.91768)

        # edm xmap x y, e(40)
        res1 = edm(t, x, y, panel=panel, E=40)
        res2 = edm(t, y, x, panel=panel, E=40)
        check_edm_results(res1, res2, 0.76444, 0.83836)

        # edm xmap x y, e(40) allowmissing
        res1 = edm(t, x, y, panel=panel, E=40, allowMissing=True)
        res2 = edm(t, y, x, panel=panel, E=40, allowMissing=True)
        check_edm_results(res1, res2, 0.63174, 0.81394)

        # edm xmap x y, e(40) idw(-1)
        res1 = edm(t, x, y, panel=panel, E=40, panelWeight=float("inf"))
        res2 = edm(t, y, x, panel=panel, E=40, panelWeight=float("inf"))
        check_edm_results(res1, res2, 0.76444, 0.83836)

        # edm xmap x y, e(40) idw(-1) allowmissing
        res1 = edm(
            t, x, y, panel=panel, E=40, panelWeight=float("inf"), allowMissing=True
        )
        res2 = edm(
            t, y, x, panel=panel, E=40, panelWeight=float("inf"), allowMissing=True
        )
        check_edm_results(res1, res2, 0.55937, 0.75815)

    def test_panel_data_with_missing_observations(self):
        obs = 100
        x, y = logistic_map(obs)
        t = np.arange(1, len(x) + 1)
        panel = (t > obs / 3).astype(int)

        # Drop some rows of the dataset & make sure the plugin can handle this
        # (i.e. can it replicate a kind of 'tsfill' hehaviour).

        # drop if mod(t,7) == 0
        x = x[t % 7 != 0]
        panel = panel[t % 7 != 0]
        t = t[t % 7 != 0]

        # edm explore x, e(5)
        res = edm(t, x, panel=panel, E=5)
        check_edm_result(res, 0.95118)

        # edm explore x, e(5) allowmissing
        res = edm(t, x, panel=panel, E=5, allowMissing=True)
        check_edm_result(res, 0.95905)

        # edm explore x, e(5) idw(-1)
        res = edm(t, x, panel=panel, E=5, panelWeight=float("inf"))
        check_edm_result(res, 0.92472)

        # edm explore x, e(5) idw(-1) allowmissing
        res = edm(t, x, panel=panel, E=5, panelWeight=float("inf"), allowMissing=True)
        check_edm_result(res, 0.93052)

        # edm explore x, e(5) idw(-1) k(-1)
        res = edm(t, x, panel=panel, E=5, panelWeight=float("inf"), k=float("inf"))
        check_edm_result(res, 0.92472)

        # See if the relative dt flags work

        # edm explore x, e(5) reldt
        res = edm(t, x, panel=panel, E=5, reldt=True)
        check_edm_result(res, 0.90239)

        # edm explore x, e(5) reldt allowmissing
        res = edm(t, x, panel=panel, E=5, reldt=True, allowMissing=True)
        check_edm_result(res, 0.9085)

        # edm explore x, e(5) idw(-1) reldt
        res = edm(t, x, panel=panel, E=5, panelWeight=float("inf"), reldt=True)
        check_edm_result(res, 0.78473)

        # edm explore x, e(5) idw(-1) reldt allowmissing
        res = edm(
            t,
            x,
            panel=panel,
            E=5,
            panelWeight=float("inf"),
            reldt=True,
            allowMissing=True,
        )
        check_edm_result(res, 0.75709)

    def test_bad_inputs(self):
        obs = 500
        x, y = logistic_map(obs)

        t = np.arange(1, len(x) + 1, dtype=float)
        x = x[299:]
        y = y[299:]
        t = t[299:]

        # Check some NA inputs don't crash R
        res = edm(t, x, y, E=None)
        assert res["rc"] == 0

        res = edm(t, x, y, E=[2, 3, None])
        assert res["rc"] == 0

        res = edm(t, x, y, k=None)
        assert res["rc"] == 0
