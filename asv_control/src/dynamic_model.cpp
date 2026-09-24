#include "asv_control/model/dynamic_model.h"

DynamicModel::DynamicModel()
    : DynamicModel(Eigen::Vector3d{0, 0, 0}, Eigen::Vector3d{0, 0, 0}) {
  ;
}

DynamicModel::DynamicModel(const Eigen::Vector3d &pose,
                           const Eigen::Vector3d &vel,
                           const ModelParams &params)
    : p(params) {
  eta = pose;
  nu = vel;
  C = Eigen::Matrix3d::Zero();
  D = Eigen::Matrix3d::Zero();
  eta_dot_last = Eigen::Vector3d::Zero();
  nu_dot_last = Eigen::Vector3d::Zero();

  M << p.m - p.X_u_dot, 0, 0,                    //
      0, p.m - p.Y_v_dot, p.m * p.xg - p.Y_r_dot,  //
      0, p.m * p.xg - p.N_v_dot, p.Iz - p.N_r_dot; //
  M_inv = M.inverse();
}

State DynamicModel::update(Azimuth u) {
  Eigen::Vector3d F;
  Eigen::Matrix<double, 3, 2> T;
  Eigen::Vector2d control;
  T << cos(u.ang0), cos(u.ang1),            //
      sin(u.ang0), sin(u.ang1),             //
      lx0 * sin(u.ang0), lx1 * sin(u.ang1); //

  control << u.force0, u.force1;
  F = T * control;
  return update_with_perturb(F, Eigen::Vector3d::Zero());
}

State DynamicModel::update(Eigen::Vector3d u) {
  return update_with_perturb(u, Eigen::Vector3d{0, 0, 0});
}

State DynamicModel::update_with_perturb(Eigen::Vector3d F,
                                        const Eigen::Vector3d &nu_c) {
  // Relative velocity vector (nu_r = nu - ocean currents)
  Eigen::Vector3d nu_r = nu - nu_c;
  DecomposedDyn dyn = get_decomposed_dyn(nu);

  // nu_dot = M.inverse() * (F - C * nu - D * nu);
  nu_dot = dyn.f + dyn.g * F;
  nu = p.integral_step * (nu_dot + nu_dot_last) / 2 + nu; // integral
  nu_dot_last = nu_dot;
  // Because of estimated coefficients, some configurations might cause the ASV
  // to diverge. Clamping the dynamics mitigates it.
  nu(0) = std::clamp(nu(0), p.max_astern, p.max_surge);
  nu(2) = std::clamp(nu(2), -p.max_yaw, p.max_yaw);

  Eigen::Matrix3d J;
  Eigen::Vector3d eta_dot;
  J << std::cos(eta(2)), -std::sin(eta(2)), 0, std::sin(eta(2)),
      std::cos(eta(2)), 0, 0, 0, 1;

  eta_dot = J * nu; // transformation into local reference frame
  eta = p.integral_step * (eta_dot + eta_dot_last) / 2 + eta; // integral
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

  return State{
      eta(0),    eta(1),    eta(2),    //
      nu(0),     nu(1),     nu(2),     //
      nu_dot(0), nu_dot(1), nu_dot(2), //
  };
}

double DynamicModel::wrap_angle(double angle) {
  double wrapped = std::fmod(angle + M_PI, 2 * M_PI);
  if (wrapped < 0)
    wrapped += 2 * M_PI;
  return wrapped - M_PI;
}

DecomposedDyn DynamicModel::get_decomposed_dyn(const Eigen::Vector3d &nu_) {
  DecomposedDyn out{};
  Eigen::Matrix3d C_RB, C_A;
  auto [surge, sway, yaw] = std::make_tuple(nu_.x(), nu_.y(), nu_.z());
  double c0 = p.m * (p.xg * yaw + sway);
  double c1 = p.m * surge;
  double c2 = p.Y_v_dot * sway + p.Y_r_dot * yaw;
  double c3 = p.X_u_dot * surge;

  C_RB << 0, 0, -c0, //
      0, 0, c1,      //
      c0, -c1, 0;    //

  C_A << 0, 0, c2, //
      0, 0, -c3,   //
      -c2, c3, 0;  //

  C = C_RB + C_A;

  auto nu_abs = nu_.cwiseAbs();
  auto [surge_abs, sway_abs, yaw_abs] =
      std::make_tuple(nu_abs.x(), nu_abs.y(), nu_abs.z());
  yaw_abs = 0.0; // Simplified for large boats (7.24)
  double d0 = -p.Xuu * surge_abs;
  double d1 = -p.Yvv * sway_abs - p.Yrv * yaw_abs;
  double d2 = -p.Yvr * sway_abs - p.Yrr * yaw_abs;
  double d3 = -p.Nvv * sway_abs - p.Nrv * yaw_abs;
  double d4 = -p.Nvr * sway_abs - p.Nrr * yaw_abs;

  D << d0, 0, 0, //
      0, d1, d2, //
      0, d3, d4; //

  out.f = -M_inv * (C * nu_ + D * nu_);
  out.g = M_inv;
  out.g_inv = M;
  return out;
}
