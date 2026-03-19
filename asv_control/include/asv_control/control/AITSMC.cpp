#include "AITSMC.h"
#include "asv_control/model/dynamic_model.h"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <cmath>
#include <iostream>

AITSMC::AITSMC() {
  e_i = Eigen::Vector3d::Zero();
  e_i_dot_last = Eigen::Vector3d::Zero();
  K = Eigen::Vector3d::Zero();
  K_dot_last = Eigen::Vector3d::Zero();
}

AITSMC::AITSMC(const AITSMCParams &params) {
  p = params;
  e_i = Eigen::Vector3d::Zero();
  e_i_dot_last = Eigen::Vector3d::Zero();
  K = Eigen::Vector3d::Zero();
  K_dot_last = Eigen::Vector3d::Zero();

  // model = DynamicModel(Eigen::Vector3d{0, 0, 0});
  beta << 0, 0, p.psi.beta;
}

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

Azimuth AITSMC::update(const State &state, const State &setpoint) {
  Eigen::Vector3d eta(state.x, state.y, state.psi);
  Eigen::Vector3d eta_d(setpoint.x, setpoint.y, setpoint.psi);
  Eigen::Vector3d nu(state.u, state.v, state.r);
  Eigen::Vector3d nu_d(setpoint.u, setpoint.v, setpoint.r);
  Eigen::Vector3d nu_dot_d(setpoint.u_dot, setpoint.v_dot, setpoint.r_dot);

  // INTEGRAL ERROR
  // e_i_dot = sign(err_n)*|err_n|^(q_n/p_n)
  Eigen::Vector3d err(nu_d(0) - nu(0), nu_d(1) - nu(1),
                      angle_dist(eta_d(2), eta(2)));
  Eigen::Vector3d qp(p.u.q / p.u.p, p.v.q / p.v.p, p.psi.q / p.psi.p);
  Eigen::Array3d qp1 = (Eigen::Vector3d::Ones() - qp).array();
  Eigen::Array3d tc(p.u.tc, p.v.tc, p.psi.tc);
  Eigen::Vector3d e_qp = err.cwiseAbs().array().pow(qp.array());
  Eigen::Vector3d e_i_dot = err.cwiseSign().cwiseProduct(e_qp);

  // INITIALIZE ALPHA
  if (!initialized) {
    alpha = (err.array().abs().pow(qp1) / (tc * qp1)).max(1e-6);
    e_i = -err.cwiseQuotient(alpha);
    initialized = true;
  }

  // SLIDING SURFACE
  // s = nu_d - nu + beta*(eta_d-eta) + alpha*e_I
  e_i = integral_step * (e_i_dot + e_i_dot_last) / 2 + e_i;
  e_i_dot_last = e_i_dot;
  Eigen::Vector3d s =
      nu_d - nu + beta.cwiseProduct(err) + alpha.cwiseProduct(e_i);

  // ADAPTIVE GAIN
  // K_dot = sqrt(K_a)*sqrt(|s|) - sqrt(K_b)*K^2
  Eigen::Vector3d K_a(p.u.k_alpha, p.v.k_alpha, p.psi.k_alpha);
  Eigen::Vector3d K_b(p.u.k_beta, p.v.k_beta, p.psi.k_beta);
  Eigen::Vector3d s_abs_sqrt = s.cwiseAbs().cwiseSqrt();
  Eigen::Vector3d K_dot = K_a.cwiseSqrt().cwiseProduct(s_abs_sqrt) -
                          K_b.cwiseProduct(K.cwiseProduct(K));
  K = integral_step * (K_dot + K_dot_last) / 2 + K;
  K_dot_last = K_dot;

  // AUXLIARY CONTROL
  Eigen::Vector3d eps(p.u.epsilon, p.v.epsilon, p.psi.epsilon);
  Eigen::Vector3d sign_s = s.cwiseSign();
  Eigen::Vector3d U_aux = -K.cwiseProduct(s_abs_sqrt).cwiseProduct(sign_s) -
                          eps.cwiseProduct(K).cwiseProduct(s.cwiseAbs());

  // DYNAMICS
  DecomposedDyn dyn = model.get_decomposed_dyn(nu);

  // CONTROL SIGNAL
  Eigen::Vector3d U =
      dyn.g_inv * (nu_dot_d - dyn.f + beta.cwiseProduct(nu_d - nu) +
                   alpha.cwiseProduct(e_i_dot) - U_aux);

  // ALLOCATE FORCES
  Azimuth out;
  double Tx = U(0);
  double Ty = U(1);
  double Tz = U(2);

  double thrust_dir = atan2(Ty, Tx); // average angle
  double Txy = std::hypot(Tx, Ty);
  double delta = atan2(Tz, model.lx0 * Txy); // split angle for yaw
  double force = Txy / (2.0 * cos(delta));

  // Clamp force
  if (force > model.u_max) {
    double scale = model.u_max / force;
    force = model.u_max;
    // Tx, Ty, Tz all scale proportionally
  }

  out.ang0 = thrust_dir + delta;
  out.ang1 = thrust_dir - delta;
  out.force0 = force;
  out.force1 = force;

  // Printing for debug
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
  std::cout << "Thrust:\n"
            << Eigen::Vector3d{Tx, 0, Tz}.format(fmt) << "\n"
            << "s:\n"
            << s.format(fmt) << "\n"
            << "U_aux:\n"
            << U_aux.format(fmt) << "\n"
            << "err:\n"
            << err.format(fmt) << "\n"
            << "e_i_dot:\n"
            << e_i_dot.format(fmt) << "\n"
            << "e_i:\n"
            << e_i.format(fmt) << "\n"
            << "alpha:\n"
            << alpha.format(fmt) << "\n"
            << std::endl;

  for (int i = 0; i < 3; i++) {
    debugData[i].e = err(i);
    debugData[i].e_i = e_i(i);
    debugData[i].e_i_dot = e_i_dot(i);
    debugData[i].s = s(i);
    debugData[i].K = K(i);
    debugData[i].U = U(i);
  }
  return out;
}

void AITSMC::reset_integral(int idx) {
  e_i(idx) = 0.0;
  e_i_dot_last(idx) = 0.0;
  K(idx) = 0.0;
  K_dot_last(idx) = 0.0;
}
