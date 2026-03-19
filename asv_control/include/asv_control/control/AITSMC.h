#pragma once
#include "../model/dynamic_model.h"
#include <array>
#include <iostream>

struct AITSMCStateParams {
  double beta, epsilon, k_alpha, k_beta, //
      tc, q, p;
};

struct AITSMCParams {
  AITSMCStateParams u, v, psi;
};

struct AITSMCDebugData {
  double e, e_i, e_i_dot, s, K, U;
};

class AITSMC {
public:
  AITSMC();
  AITSMC(const AITSMCParams &params);

  Azimuth update(const State &s, const State &setpoint);

  double normalize_angle(double angle_in);
  double angle_dist(double ang1, double ang2);
  void reset_integral(int idx);

  [[nodiscard]] AITSMCDebugData getDebugData(int idx) const {
    return debugData[idx];
  }

private:
  AITSMCParams p;

  static constexpr double integral_step{0.01};

  Eigen::Vector3d alpha;
  Eigen::Vector3d beta;
  bool initialized{false};

  Eigen::Vector3d e_i;
  Eigen::Vector3d e_i_dot_last;
  Eigen::Vector3d K;
  Eigen::Vector3d K_dot_last;
  double e_psi_last{0}, e_psi_last_last{0};

  double Ka_u{0}, Ka_psi{0};
  double ei_u{0}, ei_psi{0};

  double Ka_dot_last_u{0}, Ka_dot_last_psi{0};

  std::array<AITSMCDebugData, 3> debugData;

  DynamicModel model;
  double g_u{0}, g_psi{0};
};
