#pragma once
#include "../model/dynamic_model.h"

struct AITSMCParams {
  double beta_psi, epsilon_u, epsilon_psi, k_alpha_u, k_alpha_psi, k_beta_u,
      k_beta_psi, tc_u, tc_psi, q_u, q_psi, p_u, p_psi;
};

struct AITSMCDebugData {
  double e_u{0}, e_psi{0};
  double edot_psi{0}, ei_psi{0};
  double s_u{0}, s_psi{0};
  double Ka_u{0}, Ka_psi{0};
  double Tx{0}, Tz{0};
};

class AITSMC {
public:
  explicit AITSMC(const AITSMCParams &params);

  Azimuth update(const State &s,
                          const State &setpoint);

  static AITSMCParams defaultParams();

  double normalize_angle(double angle_in);
  double angle_dist(double ang1, double ang2);

  [[nodiscard]] AITSMCDebugData getDebugData() const { return debugData; }

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

  AITSMCDebugData debugData;

  DynamicModel model;
  double g_u{0}, g_psi{0};
};
