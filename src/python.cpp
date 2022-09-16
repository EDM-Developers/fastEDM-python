#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include <numeric>
#include <iostream>
#include <fstream>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include "cpu.h"
#include "edm.h"
#include "stats.h"

std::vector<int> bool_to_int(std::vector<bool> bv)
{
  std::vector<int> iv;
  std::copy(bv.begin(), bv.end(), std::back_inserter(iv));
  return iv;
}

#ifdef JSON
void append_to_dumpfile(std::string fName, const json& taskGroup)
{
  json allTaskGroups;

  std::ifstream i(fName);
  if (i.is_open()) {
    i >> allTaskGroups;
  }

  allTaskGroups.push_back(taskGroup);

  // Add "o << std::setw(4) << allTaskGroups" to pretty-print the saved JSON
  std::ofstream o(fName);
  o << allTaskGroups << std::endl;
}
#endif


class PythonConsoleIO : public IO
{
public:
  PythonConsoleIO() { this->verbosity = 1; }
  PythonConsoleIO(int v) { this->verbosity = v; }
  virtual void out(const char* s) const { std::cout << s; }
  virtual void error(const char* s) const { std::cerr << s; }
  virtual void flush() const { ; }
};



std::atomic<bool> isInterrupted;

bool rcpp_keep_going()
{
  return !isInterrupted;
}

void replace_nan(std::vector<double>& v)
{
  for (int i = 0; i < v.size(); i++) {
    if (!std::isnormal(v[i])) {
      v[i] = MISSING_D;
    }
  }
}


/*
Rcpp::NumericMatrix to_R_matrix(const double* v, int r, int c, std::vector<bool> filter = {}, bool rowMajor = false)
{
  Rcpp::NumericMatrix mat(r, c);

  int obsNum = 0;

  if (rowMajor) {
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        if (filter.size() > 0 && !filter[i]) {
          mat(i, j) = NA_REAL;
          continue;
        }
        mat(i, j) = v[obsNum] != MISSING_D ? v[obsNum] : NA_REAL;
        obsNum += 1;
      }
    }
  } else {
    for (int j = 0; j < c; j++) {
      for (int i = 0; i < r; i++) {
        if (filter.size() > 0 && !filter[i]) {
          mat(i, j) = NA_REAL;
          continue;
        }

        mat(i, j) = v[obsNum] != MISSING_D ? v[obsNum] : NA_REAL;
        obsNum += 1;
      }
    }
  }

  return mat;
}
*/


py::dict run_command(
    std::vector<double> t,
    std::vector<double> x,
    std::optional<std::vector<double>> y = std::nullopt,
    std::optional<std::vector<double>> copredict_x = std::nullopt,
    std::optional<std::vector<int>> panel = std::nullopt,
    std::vector<int> es = {2},
    int tau = 1,
    std::vector<double> thetas = {1.0},
    std::optional<std::vector<int>> libs = std::nullopt,
    int k = 0, std::string algorithm = "simplex",
    int numReps = 1, int p = 1, int crossfold = 0, bool full = false, bool shuffle = false,
    bool saveFinalPredictions = false, bool saveFinalCoPredictions = false,
    bool saveManifolds = false, bool saveSMAPCoeffs = false, bool dt = false, bool reldt = false,
    double dtWeight = 0.0,
    std::optional<std::vector<std::vector<double>>> extras = std::nullopt,
    bool allowMissing = false,
    double missingDistance = 0.0, double panelWeight = 0.0,
    //Rcpp::Nullable<Rcpp::NumericMatrix> panelWeights = R_NilValue, 
    int verbosity = 1, 
    bool showProgressBar = true, int numThreads = 1, bool lowMemory = false, bool predictWithPast = false, std::string saveInputs = "")
{
  try {

    PythonConsoleIO io(verbosity);
    isInterrupted = false;

    Options opts;

    opts.nthreads = numThreads;
    opts.copredict = false;
    opts.forceCompute = true;
    opts.saveManifolds = saveManifolds;
    opts.saveSMAPCoeffs = saveSMAPCoeffs;
    opts.missingdistance = missingDistance;
    opts.lowMemoryMode = lowMemory;
    opts.useOnlyPastToPredictFuture = predictWithPast;

    opts.thetas = thetas;

    std::vector<int> Es = es;

    std::vector<int> libraries;
    if (libs.has_value()) {
     libraries = libs.value();
    }

    if (algorithm == "simplex") {
      opts.algorithm = Algorithm::Simplex;
    } else if (algorithm == "smap") {
      opts.algorithm = Algorithm::SMap;
    } else {
      return {}; // TODO
    }

    opts.calcRhoMAE = true; // TODO: When is this off?
    opts.aspectRatio = 0;
    opts.distance = Distance::Euclidean;
    opts.metrics = {};
    opts.cmdLine = "";
    opts.saveKUsed = true;

    if (io.verbosity > 1) {
      io.print(fmt::format("Num threads used is {}\n", opts.nthreads));
      io.print(
      fmt::format("CPU has {} logical cores and {} physical cores\n", num_logical_cores(), num_physical_cores()));
    }

    replace_nan(t);
    replace_nan(x);
    
    // Need to wipe out this so that the default missing distance
    // calculation is fine.
    for (int i = 0; i < t.size(); i++) {
      if (t[i] == MISSING_D) {
        x[i] = MISSING_D;
      }
    }

    bool explore;
    std::vector<double> xmap;
    if (y.has_value()) {
      xmap = y.value();
      replace_nan(xmap);
      explore = false;
    } else {
      explore = true;
    }

    opts.idw = panelWeight;
    
    std::map<std::pair<int, int>, float> fakePanelWeights;

    std::vector<int> panelIDs;

    if (panel.has_value()) {
      panelIDs = panel.value();
      opts.panelMode = true;
      
      /*
      if (panelWeights.isNotNull()) {
        Rcpp::NumericMatrix matrix = Rcpp::as<Rcpp::NumericMatrix>(panelWeights);
        
        std::vector<int> uniquePanelIDs(panelIDs);
        auto it = std::unique(uniquePanelIDs.begin(), uniquePanelIDs.end());
        uniquePanelIDs.resize(std::distance(uniquePanelIDs.begin(),it));
        
        for (int i = 0; i < uniquePanelIDs.size(); i++) {
          for (int j = 0; j < uniquePanelIDs.size(); j++) {
            std::pair<int,int> key(uniquePanelIDs[i], uniquePanelIDs[j]);
            opts.idWeights[key] = Rcpp::as<Rcpp::NumericMatrix>(panelWeights)(i,j);
          }
        }

        // TODO: Perhaps throw error if both idw constant and matrix supplied.
        opts.idw = 0;
      }
      */
    } else {
      opts.panelMode = false;
    }

    std::vector<double> co_x;

    if (copredict_x.has_value()) {
      co_x = copredict_x.value();
      replace_nan(co_x);
    }

    std::vector<std::vector<double>> extrasVecs;

    if (extras.has_value()) {
      auto extrasList = extras.value();

      for (int e = 0; e < extrasList.size(); e++) {
        extrasVecs.emplace_back(extrasList[e]);
        replace_nan(extrasVecs[extrasVecs.size() - 1]);
        opts.metrics.push_back(Metric::Diff); // TODO: Handle factor extras
      }
    }

    int numExtrasLagged = 0;

    ManifoldGenerator generator(t, x, tau, p, xmap, co_x, panelIDs, extrasVecs, numExtrasLagged, dt, reldt,
                                allowMissing, dtWeight);

    if (allowMissing && opts.missingdistance == 0) {
      opts.missingdistance = default_missing_distance(x);
    }

    int maxE = Es[Es.size() - 1];
    std::vector<bool> usable = generator.generate_usable(maxE);

    int numUsable = std::accumulate(usable.begin(), usable.end(), 0);
    if (numUsable == 0) {
      io.print("Num usable is 0!\n");
      
      py::dict res("rc"_a=8000);
      return res;
    }

    if (!explore && libraries.size() == 0) {
      libraries = { numUsable };
    }

    bool copredictMode = co_x.size() > 0;
    std::string rngState = "";

#ifdef JSON
    // If requested, save the inputs to a local file for testing
    if (!saveInputs.empty()) {
      // Fill in some uninitialised Option members just so we aren't saving
      // noise to the JSON file (these will be overwritten inside edm.cpp).
      opts.numTasks = numReps * crossfold * Es.size() * (libraries.size() > 0 ? libraries.size() : 1);
      opts.aspectRatio = 1.0;
      opts.k = k;

      json taskGroup;
      taskGroup["generator"] = generator;
      taskGroup["opts"] = opts;
      taskGroup["Es"] = Es;
      taskGroup["libraries"] = libraries;
      taskGroup["k"] = k;
      taskGroup["numReps"] = numReps;
      taskGroup["crossfold"] = crossfold;
      taskGroup["explore"] = explore;
      taskGroup["full"] = full;
      taskGroup["shuffle"] = shuffle;
      taskGroup["saveFinalPredictions"] = saveFinalPredictions;
      taskGroup["saveFinalCoPredictions"] = saveFinalCoPredictions;
      taskGroup["saveSMAPCoeffs"] = saveSMAPCoeffs;
      taskGroup["copredictMode"] = copredictMode;
      taskGroup["usable"] = bool_to_int(usable);
      taskGroup["rngState"] = rngState;

      append_to_dumpfile(saveInputs, taskGroup);

      // If we just want to save the input file and not actually run the command,
      // then uncomment the following two lines to end early.
      // SF_scal_save(FINISHED_SCALAR, 1.0);
      // return SUCCESS; // Let Stata give the error here.
    }
#endif

    if (io.verbosity > 1) {
      io.print("Starting the command!\n");
      io.flush();  
    }

    auto genPtr = std::shared_ptr<ManifoldGenerator>(&generator, [](ManifoldGenerator*) {});

    std::vector<std::future<PredictionResult>> futures = launch_task_group(
      genPtr, opts, Es, libraries, k, numReps, crossfold, explore, full, shuffle, saveFinalPredictions,
      saveFinalCoPredictions, saveSMAPCoeffs, copredictMode, usable, rngState, &io, rcpp_keep_going, nullptr);

    if (io.verbosity > 1) {
      io.print(fmt::format("Waiting for {} results to come back\n", futures.size()));
      io.flush();
    }

    int rc = 0;

//    RcppThread::ProgressBar bar(futures.size(), 1);

    int kMin, kMax;

 //   Rcpp::NumericMatrix predictions, coPredictions, coeffs;
 //   Rcpp::DataFrame stats, copredStats;
 //   std::vector<Rcpp::NumericMatrix> Ms, Mps;

    {
  //    Rcpp::IntegerVector Es, libraries;
  //    Rcpp::NumericVector thetas, rhos, maes;

  //    Rcpp::IntegerVector co_Es, co_libraries;
  //    Rcpp::NumericVector co_thetas, co_rhos, co_maes;

  //    auto Rint = [](double v) { return (v != MISSING_D) ? v : NA_INTEGER; };
  //    auto Rdouble = [](double v) { return (v != MISSING_D) ? v : NA_REAL; };

      for (int f = 0; f < futures.size(); f++) {
        // TODO: Probably should check for interruptions every second
        // or so instead of after each future is completed.

        /*
        isInterrupted = RcppThread::isInterrupted();

        if (isInterrupted) {
          Rcpp::List res;
          res["rc"] = 1;
          return res;
        }
        */

        const PredictionResult pred = futures[f].get();

        /*
        if (showProgressBar) {
          bar++;
        }
        */

        if (f == 0 || pred.kMin < kMin) {
          kMin = pred.kMin;
        }
        if (f == 0 || pred.kMax > kMax) {
          kMax = pred.kMax;
        }

        /*
        if (!pred.copredict) {
          for (int t = 0; t < pred.stats.size(); t++) {
            Es.push_back(Rint(pred.stats[t].E));
            thetas.push_back(Rdouble(pred.stats[t].theta));
            libraries.push_back(Rint(pred.stats[t].library));
            rhos.push_back(Rdouble(pred.stats[t].rho));
            maes.push_back(Rdouble(pred.stats[t].mae));
          }
        } else {
          for (int t = 0; t < pred.stats.size(); t++) {
            co_Es.push_back(Rint(pred.stats[t].E));
            co_thetas.push_back(Rdouble(pred.stats[t].theta));
            co_libraries.push_back(Rint(pred.stats[t].library));
            co_rhos.push_back(Rdouble(pred.stats[t].rho));
            co_maes.push_back(Rdouble(pred.stats[t].mae));
          }
        }
        */

        if (pred.rc > rc) {
          rc = pred.rc;
        }

/*
        if (pred.predictions != nullptr) {
          if (!pred.copredict) {
            predictions =
              to_R_matrix(pred.predictions.get(), pred.predictionRows.size(), pred.numThetas, pred.predictionRows);
          } else {
            coPredictions =
              to_R_matrix(pred.predictions.get(), pred.predictionRows.size(), pred.numThetas, pred.predictionRows);
          }
        }
        if (pred.coeffs != nullptr) {
          coeffs = to_R_matrix(pred.coeffs.get(), pred.predictionRows.size(), pred.numCoeffCols, pred.predictionRows);
        }

        if (saveManifolds) {
          Ms.push_back(to_R_matrix(pred.M->data(), pred.M->numPoints(), pred.M->E_actual(), {}, true));
          Mps.push_back(to_R_matrix(pred.Mp->data(), pred.Mp->numPoints(), pred.Mp->E_actual(), {}, true));
        }
      }

      stats = Rcpp::DataFrame::create(Rcpp::_["E"] = Es, Rcpp::_["library"] = libraries, Rcpp::_["theta"] = thetas,
                                        Rcpp::_["rho"] = rhos, Rcpp::_["mae"] = maes);

      if (copredictMode) {
        copredStats =
          Rcpp::DataFrame::create(Rcpp::_["E"] = co_Es, Rcpp::_["library"] = co_libraries, Rcpp::_["theta"] = co_thetas,
                                  Rcpp::_["rho"] = co_rhos, Rcpp::_["mae"] = co_maes);
      }
*/
    }
    }

    py::dict res;

    res["rc"] = rc;

    /*
    res["stats"] = stats;
    res["kMin"] = kMin;
    res["kMax"] = kMax;

    if (copredictMode) {
      res["copredStats"] = copredStats;
    }

    if (saveFinalPredictions) {
      res["predictions"] = predictions;
    }

    if (saveFinalCoPredictions) {
      res["copredictions"] = coPredictions;
    }

    if (saveManifolds) {
      res["Ms"] = Ms;
      res["Mps"] = Mps;
    }

    if (saveSMAPCoeffs) {
      res["coeffs"] = coeffs;
    }

    if (allowMissing) {
      res["missingdistance"] = opts.missingdistance;
    }

    if (dt || reldt) {
      res["dtWeight"] = generator.dtWeight();
    }
    */

    return res;
  } catch (const std::exception& e) {
    //Rcpp::Rcerr << e.what() << std::endl;
  } catch (...) {
    //Rcpp::Rcerr << "Unknown error in the C++ code of edm" << std::endl;
  }

  py::dict res;
  res["rc"] = 8000;
  return res;
}

PYBIND11_MODULE(fastEDM, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", [](int i, int j) { return i + j; }, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    /*
    std::vector<double> t,
    std::vector<double> x,
    std::optional<std::vector<double>> y = std::nullopt,
    std::optional<std::vector<double>> copredict_x = std::nullopt,
    std::optional<std::vector<int>> panel = std::nullopt,
    std::vector<int> es = {2},
    int tau = 1,
    std::vector<double> thetas = {1.0},
    std::optional<std::vector<int>> libs = std::nullopt,
    int k = 0, std::string algorithm = "simplex",
    int numReps = 1, int p = 1, int crossfold = 0, bool full = false, bool shuffle = false,
    bool saveFinalPredictions = false, bool saveFinalCoPredictions = false,
    bool saveManifolds = false, bool saveSMAPCoeffs = false, bool dt = false, bool reldt = false,
    double dtWeight = 0.0,
    std::optional<std::vector<std::vector<double>> extras = std::nullopt,
    bool allowMissing = false,
    double missingDistance = 0.0, double panelWeight = 0.0,
    //Rcpp::Nullable<Rcpp::NumericMatrix> panelWeights = R_NilValue, 
    int verbosity = 1, 
    bool showProgressBar = true, int numThreads = 1, bool lowMemory = false, bool predictWithPast = false, std::string saveInputs = "")
    */
    m.def("run_command", run_command, py::arg("t"), py::arg("x"),
        "y"_a=std::nullopt, "copredict_x"_a=std::nullopt, "panel"_a=std::nullopt,
        "es"_a=std::vector<int>({2}), "tau"_a=1, "thetas"_a=std::vector<double>({1.0}),
        "libs"_a=std::nullopt, "k"_a=0, "algorithm"_a="simplex", "numReps"_a=1, "p"_a=1, "crossfold"_a=0,
        "full"_a=false, "shuffle"_a=false, "saveFinalPredictions"_a=false,
        "saveFinalCoPredictions"_a=false, "saveManifolds"_a=false, "saveSMAPCoeffs"_a=false,
        "dt"_a=false, "reldt"_a=false, "dtWeight"_a=0.0, "extras"_a=std::nullopt, "allowMissing"_a=false,
        "missingDistance"_a=0.0, "panelWeight"_a=0.0, "verbosity"_a=1, "showProgressBar"_a=true,
        "numThreads"_a=1, "lowMemory"_a=false, "predictWithPast"_a=false,
        "saveInputs"_a="",
     R"pbdoc(
        Run an EDM command
        A really long description...
    )pbdoc");

    m.attr("__version__") = "dev";
}
