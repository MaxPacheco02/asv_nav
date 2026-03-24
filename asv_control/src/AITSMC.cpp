#include "asv_control/control/AITSMC.h"

AITSMC::AITSMC() = default;

double AITSMC::normalize_angle(double angle_in) {
  double angle_out = std::fmod(angle_in + M_PI, 2 * M_PI);
  if (angle_out < 0) {
    angle_out += 2 * M_PI;
  }
  return angle_out - M_PI;
}

double AITSMC::angle_dist(double ang1, double ang2) {
  double diff = ang1 - ang2;
  return normalize_angle(diff);
}

Eigen::Matrix3d AITSMC::rotation_matrix(double ang) {
  Eigen::Matrix3d out;
  out << cos(ang), -sin(ang), 0, //
      sin(ang), cos(ang), 0,     //
      0, 0, 1;
  return out;
}

Eigen::Matrix3d AITSMC::rotation_matrix_dot(double ang, double r) {
  Eigen::Matrix3d out;
  out << -sin(ang), -cos(ang), 0, //
      cos(ang), -sin(ang), 0,     //
      0, 0, 0;
  return r * out;
}

void AITSMC::reset_integral(int idx) {
  e_i(idx) = 0.0;
  e_i_dot_last(idx) = 0.0;
  K(idx) = 0.0;
  K_dot_last(idx) = 0.0;
  initialized = false;
}

void AITSMC::reset_integral() {
  e_i = Eigen::Vector3d::Zero();
  e_i_dot_last = Eigen::Vector3d::Zero();
  K = Eigen::Vector3d::Zero();
  K_dot_last = Eigen::Vector3d::Zero();
  initialized = false;
}
