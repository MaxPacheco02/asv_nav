#ifndef AITSMC_H
#define AITSMC_H

#include "asv_control/model/dynamic_model.h"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

struct AITSMCStateParams {
  double beta, epsilon, k_alpha, k_beta, tc, q, p;
};

struct AITSMCDebugData {
  double e, e_i, e_i_dot, s, K, U;
};

class AITSMC {
public:
  AITSMC();

  static Eigen::Matrix3d rotation_matrix(double ang);
  static Eigen::Matrix3d rotation_matrix_dot(double ang, double r);
  static double normalize_angle(double angle_in);
  static double angle_dist(double ang1, double ang2);
  void reset_integral();
  void reset_integral(int idx);

  [[nodiscard]] AITSMCDebugData getDebugData(int idx) const {
    return debugData[idx];
  }

protected:
  static constexpr double integral_step{0.01};

  Eigen::Vector3d alpha{Eigen::Vector3d::Zero()};
  Eigen::Vector3d beta{Eigen::Vector3d::Zero()};
  bool initialized{false};

  Eigen::Vector3d e_i{Eigen::Vector3d::Zero()};
  Eigen::Vector3d e_i_dot_last{Eigen::Vector3d::Zero()};
  Eigen::Vector3d K{Eigen::Vector3d::Zero()};
  Eigen::Vector3d K_dot_last{Eigen::Vector3d::Zero()};

  std::array<AITSMCDebugData, 3> debugData;
  DynamicModel model;
};

#endif
