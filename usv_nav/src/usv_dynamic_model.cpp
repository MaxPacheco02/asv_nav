#include "usv_nav/usv_dynamic_model.h"

#include <algorithm>
#include <cmath>

ModelParams UsvDynamicModel::usv_params() {
  ModelParams p;
  p.m = 30.0;
  p.xg = 0.0;
  p.Iz = 4.1;

  // Added mass
  p.X_u_dot = -2.25;
  p.Y_v_dot = -23.13;
  p.Y_r_dot = -1.31;
  p.N_v_dot = -16.41;
  p.N_r_dot = -2.79;

  // Nonlinear damping
  p.Xuu = -70.92;
  p.Yvv = -99.99;
  p.Yvr = -5.49;
  p.Yrv = -5.49;
  p.Yrr = -8.8;
  p.Nvv = -5.49;
  p.Nvr = -8.8;
  p.Nrv = -8.8;
  p.Nrr = -3.49;

  // Divergence guards, looser than the MPC state bounds
  p.max_surge = 2.5;
  p.max_astern = -1.5;
  p.max_yaw = 3.0;
  return p;
}

UsvDynamicModel::UsvDynamicModel(const Eigen::Vector3d &pose,
                                 const Eigen::Vector3d &vel)
    : DynamicModel(pose, vel, usv_params()) {}

State UsvDynamicModel::update(double T_port, double T_stbd) {
  double Tp = std::clamp(T_port, T_min, T_max);
  double Ts = std::clamp(T_stbd, T_min, T_max);
  return DynamicModel::update(
      Eigen::Vector3d{Tp + Ts, 0.0, 0.5 * B * (Tp - Ts)});
}

std::pair<double, double> UsvDynamicModel::allocate(double X, double N) {
  // X = Tp + Ts, N = 0.5 * B * (Tp - Ts)
  double Tp = X / 2.0 + N / B;
  double Ts = X / 2.0 - N / B;

  // Scale both thrusts equally so the X/N ratio (heading intent) is kept
  auto limit = [](double T) { return T >= 0 ? T_max / T : T_min / T; };
  double scale = std::min({1.0, limit(Tp), limit(Ts)});
  return {Tp * scale, Ts * scale};
}

DecomposedDyn UsvDynamicModel::get_decomposed_dyn(const Eigen::Vector3d &nu_) {
  DecomposedDyn out{};
  Eigen::Matrix3d C_RB, C_A;
  auto [surge, sway, yaw] = std::make_tuple(nu_.x(), nu_.y(), nu_.z());
  double c0 = p.m * (p.xg * yaw + sway);
  double c1 = p.m * surge;
  double c2 = 2.0 * (p.Y_v_dot * sway + 0.5 * (p.Y_r_dot + p.N_v_dot) * yaw);
  double c3 = p.X_u_dot * p.m * surge;

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
  double vel = std::sqrt(surge * surge + sway * sway + 1e-9);

  // Linear damping. Surge drag switches regime above 1.2 m/s.
  double Xu = surge > 1.2 ? 64.55 : -25.0;
  double Xuu = surge > 1.2 ? p.Xuu : 0.0;
  double k = 0.09 * 0.09 * 1.01;
  double Yv = 0.5 * (-40000.0 * sway_abs) *
              (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) +
               0.016 * std::pow(0.27 / 0.09, 2));
  double Yr = 6.0 * (-M_PI * 1000.0) * vel * k;
  double Nv = 0.06 * (-M_PI * 1000.0) * vel * k;
  double Nr = 0.02 * (-M_PI * 1000.0) * vel * k * 1.01;

  D << -Xu - Xuu * surge_abs, 0, 0,                                     //
      0, -Yv - (p.Yvv * sway_abs + p.Yvr * yaw_abs),                    //
      -Yr - (p.Yrv * sway_abs + p.Yrr * yaw_abs),                       //
      0, -Nv - (p.Nvv * sway_abs + p.Nvr * yaw_abs),                    //
      -Nr - (p.Nrv * sway_abs + p.Nrr * yaw_abs);                       //

  out.f = -M_inv * (C * nu_ + D * nu_);
  out.g = M_inv;
  out.g_inv = M;
  return out;
}
