#ifndef DYNAMIC_MODEL_H
#define DYNAMIC_MODEL_H

#include <Eigen/Dense>
#include <iostream>

struct State {
  double x, y, psi;
  double u, v, r;
};

class DynamicModel {
 public:
  DynamicModel();
  DynamicModel(const Eigen::Vector3d& pose);

  State update(double left_thruster, double right_thruster);
  State update_with_perturb(double left_thruster, double right_thruster,
                            const Eigen::Vector3d& nu_c);
  static double wrap_angle(double angle);

 private:
  // Model parameters
  constexpr static double m = 4725629.25,  // [Kg]
      Lpp = 137.2,                         // Length between perpendiculars [m]
      xg = 0,                              // Length from o_b to CG along x-axis
      X_u_dot = 187765,                    // Added masses param
      Y_v_dot = 3780505,                   // Added masses param
      Y_r_dot = 0,                         // Added masses param
      N_r_dot = 1748950469,                // Added masses param
      Iz = 5829430000,                     // Moment of inertia
      B = 20.0;

  // Damping coefficients
  constexpr static double Xuu = -7057.485120, Yvv = -3890570.407734,
                          Yrv = -380816892.435056, Yvr = -11193515.732379,
                          Yrr = -16112020985.006985, Nvv = -138515283.493993,
                          Nrv = -8148922936.922905, Nvr = -3491938401.993457,
                          Nrr = -390901820542.178894;

  // Update rate
  constexpr static double integral_step = 0.01;

  // State
  Eigen::Vector3d eta = Eigen::Vector3d::Zero();
  Eigen::Vector3d eta_dot_last = Eigen::Vector3d::Zero();
  Eigen::Vector3d nu = Eigen::Vector3d::Zero();
  Eigen::Vector3d nu_dot = Eigen::Vector3d::Zero();
  Eigen::Vector3d nu_dot_last = Eigen::Vector3d::Zero();
  Eigen::Matrix3d M, C, D;
};

#endif  // DYNAMIC_MODEL_H
