#include "dynamic_model.h"

DynamicModel::DynamicModel() : DynamicModel(Eigen::Vector3d{0, 0, 0}) { ; }

DynamicModel::DynamicModel(const Eigen::Vector3d& pose) {
  eta = pose;
  nu = Eigen::Vector3d::Zero();
  C = Eigen::Matrix3d::Zero();
  D = Eigen::Matrix3d::Zero();
  eta_dot_last = Eigen::Vector3d::Zero();
  nu_dot_last = Eigen::Vector3d::Zero();

  M << m - X_u_dot, 0, 0,                 //
      0, m - Y_v_dot, m * xg - Y_r_dot,   //
      0, m * xg - Y_r_dot, Iz - N_r_dot;  //
}

State DynamicModel::update(Azimuth u) {
  return update_with_perturb(u, Eigen::Vector3d{0, 0, 0});
}

State DynamicModel::update_with_perturb(Azimuth u,
                                        const Eigen::Vector3d& nu_c) {
  Eigen::Matrix3d C_RB, C_A;
  Eigen::Vector3d nu_r = nu - nu_c;
  auto [surge, sway, yaw] = std::make_tuple(nu_r.x(), nu_r.y(), nu_r.z());
  double c0 = m * (xg * yaw + sway);
  double c1 = m * surge;
  double c2 = Y_v_dot * sway + Y_r_dot * yaw;
  double c3 = X_u_dot * surge;

  C_RB << 0, 0, -c0,  //
      0, 0, c1,       //
      c0, -c1, 0;     //

  C_A << 0, 0, c2,  //
      0, 0, -c3,    //
      -c2, c3, 0;   //

  C = C_RB + C_A;

  // Relative velocity vector (nu_r = nu - ocean currents)
  auto nu_abs = nu_r.cwiseAbs();
  auto [surge_abs, sway_abs, yaw_abs] =
      std::make_tuple(nu_abs.x(), nu_abs.y(), nu_abs.z());
  double d0 = -Xuu * surge_abs;
  double d1 = -Yvv * sway_abs - Yrv * yaw_abs;
  double d2 = -Yvr * sway_abs - Yrr * yaw_abs;
  double d3 = -Nvv * sway_abs - Nrv * yaw_abs;
  double d4 = -Nvr * sway_abs - Nrr * yaw_abs;
  D << d0, 0, 0,  //
      0, d1, d2,  //
      0, d3, d4;  //

  Eigen::Vector3d F;
  Eigen::Matrix<double, 3, 2> T;
  Eigen::Vector2d control;
  T << cos(u.ang0), cos(u.ang1),             //
      sin(u.ang0), sin(u.ang1),              //
      lx0 * sin(u.ang0), lx1 * sin(u.ang1);  //
  control << u.force0, u.force1;
  F = T*control;

  nu_dot = M.inverse() * (F - C * nu_r - D * nu_r);
  nu = integral_step * (nu_dot + nu_dot_last) / 2 + nu;  // integral
  nu_dot_last = nu_dot;

  Eigen::Matrix3d J;
  Eigen::Vector3d eta_dot;
  J << std::cos(eta(2)), -std::sin(eta(2)), 0, std::sin(eta(2)),
      std::cos(eta(2)), 0, 0, 0, 1;

  eta_dot = J * nu;  // transformation into local reference frame
  eta = integral_step * (eta_dot + eta_dot_last) / 2 + eta;  // integral
  eta_dot_last = eta_dot;

  // Printing for debug
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
  std::cout << "M:\n"
            << M.format(fmt) << "\n"
            << "C:\n"
            << C.format(fmt) << "\n"
            << "D:\n"
            << D.format(fmt) << "\n"
            << "T:\n"
            << F.format(fmt) << "\n"
            << "nu:\n"
            << nu.format(fmt) << "\n"
            << "nu_dot:\n"
            << nu_dot.format(fmt) << "\n"
            << "eta:\n"
            << eta.format(fmt) << "\n"
            << std::endl;

  return State{eta(0), eta(1), eta(2), nu(0), nu(1), nu(2)};
}

double DynamicModel::wrap_angle(double angle) {
  double wrapped = std::fmod(angle + M_PI, 2 * M_PI);
  if (wrapped < 0) wrapped += 2 * M_PI;
  return wrapped - M_PI;
}