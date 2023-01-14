#pragma once

/* Placeholders for missing values */
#include <limits> // std::numeric_limits

const double MISSING_D = 1.0e+100;
const float MISSING_F = 1.0e+30;
constexpr int MISSING_I = std::numeric_limits<int>::min();

#include <memory>
#include <utility>
#include <vector>

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

#ifdef JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

#if defined(WITH_ARRAYFIRE)
#include <arrayfire.h>

struct ManifoldOnGPU
{
  af::array mdata;   // shape [_E_actual _numPoints 1 1] - manifold
  af::array targets; // Shape [_numPoints 1 1 1]
  af::array panel;   // Shape [_numPoints 1 1 1] - panel ids
  int numPoints, E_x, E_dt, E_extras, E_lagged_extras, E_actual;
  double missing;
};
#endif

#include "stats.h"

class ManifoldGenerator
{
private:
  std::vector<double> _t, _x, _xmap, _co_x;
  std::vector<std::vector<double>> _extras;
  std::vector<int> _panelIDs;
  int _tau, _p, _num_extras, _num_extras_lagged;
  bool _dt, _reldt, _allow_missing, _panel_mode, _xmap_mode;
  std::vector<int> _observation_number;
  double _dtWeight;

  void setup_observation_numbers();

  bool find_observation_num(int target, int& k, int direction, int panel) const;

  double default_dt_weight()
  {
    auto xObserved = remove_value<double>(_x, MISSING_D);
    double xSD = standard_deviation(xObserved);

    std::vector<double> dts = this->dts();
    auto dtObserved = remove_value<double>(dts, MISSING_D);
    double dtSD = standard_deviation(dtObserved);

    if (dtSD == 0.0) {
      return -1;
    } else {
      return xSD / dtSD;
    }
  }

  double get_dt(int i) const;

public:
  void fill_in_point(int i, int E, bool copredictionMode, bool predictionSet, double dtWeight, double* point) const;
  double get_target(int i, bool copredictionMode, bool predictionSet, int& targetIndex) const;

  double calculate_time_increment() const;
  int get_observation_num(int i) const { return _observation_number[i]; }

#ifdef JSON
  friend void to_json(json& j, const ManifoldGenerator& g);
  friend void from_json(const json& j, ManifoldGenerator& g);
#endif

  ManifoldGenerator() = default;

  ManifoldGenerator(const std::vector<double>& t, const std::vector<double>& x, int tau, int p,
                    const std::vector<double>& xmap = {}, const std::vector<double>& co_x = {},
                    const std::vector<int>& panelIDs = {}, const std::vector<std::vector<double>>& extras = {},
                    int numExtrasLagged = 0, bool dt = false, bool reldt = false, bool allowMissing = false,
                    double dtWeight = 0)
    : _t(t)
    , _x(x)
    , _xmap(xmap)
    , _co_x(co_x)
    , _extras(extras)
    , _panelIDs(panelIDs)
    , _tau(tau)
    , _p(p)
    , _num_extras((int)extras.size())
    , _num_extras_lagged(numExtrasLagged)
    , _dt(dt)
    , _reldt(reldt)
    , _allow_missing(allowMissing)
    , _panel_mode(panelIDs.size() > 0)
    , _xmap_mode(xmap.size() > 0)
    , _dtWeight(dtWeight)
  {
    if ((dt || reldt) && dtWeight == 0.0) {
      // If we have to set the default 'dt' weight, then make a manifold with dtweight of 1 then
      // we can rescale this by the appropriate variances in the future.
      _dt = true;
      _reldt = false;
      setup_observation_numbers();
      double defaultDTWeight = default_dt_weight();

      if (defaultDTWeight > 0) {
        _dt = dt;
        _reldt = reldt;
        _dtWeight = defaultDTWeight;
      } else {
        _dt = false;
        _reldt = false;
        _dtWeight = 0;
      }
    }

    setup_observation_numbers();
  }

  double dtWeight() const { return _dtWeight; }
  int numObs() const { return _t.size(); }
  double time(int i) const { return _t[i]; }
  bool panelMode() const { return _panel_mode; }
  int panel(int i) const { return _panelIDs[i]; }
  const std::vector<int>& panelIDs() const { return _panelIDs; }

  std::vector<bool> generate_usable(int maxE, bool copredictionMode = false) const;

  std::vector<double> dts() const
  {
    // If the user turns on dt mode but doesn't specify a weight for the time differenced variables, then
    // this method is used to generate all the time differences for the dataset so that they can be normalised.
    // Alternatively, when the user requests to save the vector of the time differences for the dataset, this is called.
    // As the first dt column is a bit special, this function returns the second dt column in the manifold.
    std::vector<double> dts;
    for (int i = 0; i < _t.size(); i++) {
      dts.push_back(get_dt(i));
    }
    return dts;
  }

  int E_dt(int E) const { return (_dt || _reldt) * E; }
  int E_extras(int E) const { return _num_extras + _num_extras_lagged * (E - 1); }
  int E_actual(int E) const { return E + E_dt(E) + E_extras(E); }

  int numExtrasLagged() const { return _num_extras_lagged; }
  int numExtras() const { return _num_extras; }
};

class Manifold
{
  std::shared_ptr<const ManifoldGenerator> _gen;
  std::unique_ptr<double[]> _flat = nullptr;
  std::vector<double> _targets;
  std::vector<double> _targetTimes;
  std::vector<double> _pointTimes;
  std::vector<int> _panelIDs;
  std::vector<int> _pointNumToStartIndex;
  int _numPoints, _E_x, _E_dt, _E_extras, _E_lagged_extras, _E_actual;
  bool _predictionSet, _copredictMode;
  double _dtWeight;

  void init(int E, const std::vector<bool>& filter, bool predictionSet, bool copredictMode, bool lazy)
  {
    _E_x = E;
    _E_dt = _gen->E_dt(E);
    _E_extras = _gen->E_extras(E);
    _E_lagged_extras = _gen->numExtrasLagged() * E;
    _E_actual = _gen->E_actual(E);

    bool takeEveryPoint = filter.size() == 0;

    for (int i = 0; i < _gen->numObs(); i++) {
      if (takeEveryPoint || filter[i]) {

        // Throwing away library set points whose targets are missing.
        int targetIndex = i;
        double target = _gen->get_target(i, copredictMode, predictionSet, targetIndex);
        if (!predictionSet && (target == MISSING_D)) {
          continue;
        }

        _pointNumToStartIndex.push_back(i);
        _pointTimes.push_back(_gen->time(i));

        _targets.push_back(target);
        _targetTimes.push_back(target != MISSING_D ? _gen->time(targetIndex) : MISSING_D);

        if (_gen->panelMode()) {
          _panelIDs.push_back(_gen->panel(i));
        }
      }
    }

    _numPoints = _pointNumToStartIndex.size();

    if (!lazy) {
      keenly_generate_manifold();
    }
  }

  void keenly_generate_manifold()
  {
    // Fill in the manifold row-by-row (point-by-point)
    _flat = std::unique_ptr<double[]>(new double[_numPoints * _E_actual], std::default_delete<double[]>());

    for (int i = 0; i < _numPoints; i++) {
      double* point = &(_flat[i * _E_actual]);
      _gen->fill_in_point(_pointNumToStartIndex[i], _E_x, _copredictMode, _predictionSet, _dtWeight, point);
    }
  }

public:
  Manifold(const std::shared_ptr<ManifoldGenerator> gen, int E, const std::vector<bool>& filter, bool predictionSet,
           bool copredictMode = false, bool lazy = false)
    : _gen(gen)
    , _predictionSet(predictionSet)
    , _copredictMode(copredictMode)
    , _dtWeight(gen->dtWeight())
  {
    init(E, filter, predictionSet, copredictMode, lazy);
  }

  Manifold(const ManifoldGenerator& gen, int E, const std::vector<bool>& filter, bool predictionSet,
           bool copredictMode = false, bool lazy = false)
    : _predictionSet(predictionSet)
    , _copredictMode(copredictMode)
    , _dtWeight(gen.dtWeight())
  {
    _gen = std::shared_ptr<const ManifoldGenerator>(&gen, [](const ManifoldGenerator*) {});
    init(E, filter, predictionSet, copredictMode, lazy);
  }

  double operator()(int i, int j) const { return _flat[i * _E_actual + j]; }

  void eager_fill_in_point(int i, double* point) const
  {
    for (int j = 0; j < _E_actual; j++) {
      point[j] = _flat[i * _E_actual + j];
    }
  }

  void lazy_fill_in_point(int i, double* point) const
  {
    _gen->fill_in_point(_pointNumToStartIndex[i], _E_x, _copredictMode, _predictionSet, _dtWeight, point);
  }

  Eigen::Map<const Eigen::VectorXd> targetsMap() const { return { &(_targets[0]), _numPoints }; }

  double x(int i, int j) const { return this->operator()(i, j); }
  double dt(int i, int j) const { return this->operator()(i, _E_x + j); }
  double extras(int i, int j) const { return this->operator()(i, _E_x + _E_dt + j); }

  double targetTime(int i) const { return _targetTimes[i]; }
  double pointTime(int i) const { return _pointTimes[i]; }
  int panel(int i) const { return _panelIDs[i]; }

  double missing() const { return MISSING_D; }

  double target(int i) const { return _targets[i]; }
  int numTargets() const { return (int)_targets.size(); }
  const std::vector<double>& targets() const { return _targets; }

  double* data() const { return _flat.get(); };

  int numPoints() const { return _numPoints; }
  int E() const { return _E_x; }
  int E_dt() const { return _E_dt; }
  int E_lagged_extras() const { return _E_lagged_extras; }
  int E_extras() const { return _E_extras; }
  int E_actual() const { return _E_actual; }
  const std::vector<int>& panelIDs() const { return _panelIDs; }

#if defined(WITH_ARRAYFIRE)
  ManifoldOnGPU toGPU(const bool useFloat = false) const;
#endif
};
