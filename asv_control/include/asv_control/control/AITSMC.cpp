#include "AITSMC.h"
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <algorithm>
#include <iostream>

AITSMC::AITSMC(const AITSMCParams &params) {
  p = params;
  e_i = Eigen::Vector3d::Zero();
  e_i_dot_last = Eigen::Vector3d::Zero();
  K = Eigen::Vector3d::Zero();
  K_dot_last = Eigen::Vector3d::Zero();

  beta << 0, 0, p.beta_psi;
}

double AITSMC::normalize_angle(double angle_in){
  double angle_out = std::fmod(angle_in + M_PI, 2 * M_PI);
  if(angle_out < 0){
    angle_out += 2*M_PI;
  }
  return angle_out - M_PI;
}

double AITSMC::angle_dist(double ang1, double ang2){
  double diff = ang1 - ang2;
  return normalize_angle(diff);
}

Azimuth AITSMC::update(const State &state, const State &setpoint) {
  Eigen::Vector3d eta(state.x, state.y, state.psi);
  Eigen::Vector3d eta_d(setpoint.x, setpoint.y, setpoint.psi);
  Eigen::Vector3d nu(state.u, state.v, state.r);
  Eigen::Vector3d nu_d(setpoint.u, setpoint.v, setpoint.r);
  Eigen::Vector3d nu_dot(state.u_dot, state.v_dot, state.r_dot);
  Eigen::Vector3d nu_dot_d(setpoint.u_dot, setpoint.v_dot, setpoint.r_dot);

  // INTEGRAL ERROR
  // e_i_dot = sign(err_n)*|err_n|^(q_n/p_n)
  Eigen::Vector3d err(nu_d(0)-nu(0), 0, angle_dist(eta_d(2),eta(2)));
  Eigen::Vector3d qp(p.q_u/p.p_u, 0, p.q_psi/p.p_psi);
  Eigen::Vector3d e_qp = err.cwiseAbs().array().pow(qp.array());
  Eigen::Vector3d e_i_dot = err.cwiseSign().cwiseProduct(e_qp);

  // INITIALIZE ALPHA
  if (!initialized) {
    double qp_u = 1.0 - p.q_u / p.p_u;
    double qp_psi = 1.0 - p.q_psi / p.p_psi;
    alpha(0) =
        std::max(std::pow(std::abs(err(0)), qp_u) / (p.tc_u * qp_u), 1e-6);
    alpha(1) = 0;
    alpha(2) = std::max(
        std::pow(std::abs(err(2)), qp_psi) / (p.tc_psi * qp_psi), 1e-6);
    e_i = -err.cwiseQuotient(alpha);
    initialized = true;
  }

  // SLIDING SURFACE
  // s = nu_d - nu + beta*(eta_d-eta) + alpha*e_I
  e_i = integral_step * (e_i_dot + e_i_dot_last) / 2 + e_i;
  e_i_dot_last = e_i_dot;
  Eigen::Vector3d s =
      nu_d - nu + beta.cwiseProduct(eta_d - eta) + alpha.cwiseProduct(e_i);

  // ADAPTIVE GAIN
  // K_dot = sqrt(K_a)*sqrt(|s|) - sqrt(K_b)*K^2
  Eigen::Vector3d K_a(p.k_alpha_u, 0, p.k_alpha_psi);
  Eigen::Vector3d K_b(p.k_beta_u, 0, p.k_beta_psi);
  Eigen::Vector3d s_abs_sqrt = s.cwiseAbs().cwiseSqrt();
  Eigen::Vector3d K_dot = K_a.cwiseSqrt().cwiseProduct(s_abs_sqrt) -
                          K_b.cwiseProduct(K.cwiseProduct(K));
  K = integral_step * (K_dot + K_dot_last) / 2 + K;
  K_dot_last = K_dot;

  // AUXLIARY CONTROL
  Eigen::Vector3d eps(p.epsilon_u, 0, p.epsilon_psi);
  Eigen::Vector3d sign_s = s.cwiseSign();
  Eigen::Vector3d U_aux = -K.cwiseProduct(s_abs_sqrt).cwiseProduct(sign_s) -
                          eps.cwiseProduct(K).cwiseProduct(s.cwiseAbs());

  // DYNAMICS
  DecomposedDyn dyn = model.get_decomposed_dyn(nu);

  // CONTROL SIGNAL
  Eigen::Vector3d U = (nu_dot_d - dyn.f + beta.cwiseProduct(nu_d - nu) +
                       alpha.cwiseProduct(e_i_dot) - U_aux)
                          .cwiseQuotient(dyn.g);

  // ALLOCATE FORCES
  Azimuth out;
  double Tx = U(0);
  double Tz = U(2);

  double angle = atan2(Tz, Tx * model.lx0);

  out.ang0 = angle;
  out.ang1 = -angle;
  out.force0 = std::hypot(Tx / 2, Tz / model.lx0);
  out.force1 = std::hypot(Tx / 2, Tz / model.lx0);
  return out;
}
