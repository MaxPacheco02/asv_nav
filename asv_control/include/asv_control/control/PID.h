#ifndef PID_H
#define PID_H

#include "asv_control/model/dynamic_model.h"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

struct PIDStateParams {
  double p, i, d, i_max, d_alpha;
};

struct PIDDebugData {
  double e, kP, kI, kD, U;
};

struct PIDParams {
  PIDStateParams u, v, r;
};

class PID {
public:
  PID();
  PID(const PIDParams &params);

  static Eigen::Matrix3d rotation_matrix(double ang);
  static Eigen::Matrix3d rotation_matrix_dot(double ang, double r);
  static double normalize_angle(double angle_in);
  static double angle_dist(double ang1, double ang2);
  Azimuth update(const State &s, const State &setpoint);

  [[nodiscard]] PIDDebugData getDebugData(int idx) const {
    return debugData[idx];
  }

private:
  static constexpr double integral_step{0.01};
  PIDParams p;
  bool initialized{false};
  Eigen::Vector3d err_i{Eigen::Vector3d::Zero()};
  Eigen::Vector3d err_last{Eigen::Vector3d::Zero()};
  Eigen::Vector3d nu_last{Eigen::Vector3d::Zero()};
  Eigen::Vector3d err_d_last{Eigen::Vector3d::Zero()};
  Eigen::Vector3d kP, kI, kD, kI_max, kD_alpha;

  std::array<PIDDebugData, 3> debugData;
  DynamicModel model;

  double prev_ang0{0.0};
  double prev_ang1{0.0};
};

#endif
