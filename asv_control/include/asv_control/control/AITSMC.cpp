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

  beta << p.x.beta, p.y.beta, p.psi.beta;
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

Azimuth AITSMC::update(const State &state, const State &setpoint) {
  Eigen::Matrix3d J = rotation_matrix(state.psi);
  Eigen::Matrix3d J_inv = J.transpose();
  Eigen::Matrix3d J_dot = rotation_matrix_dot(state.psi, state.r);
  Eigen::Vector3d eta(state.x, state.y, state.psi);
  Eigen::Vector3d eta_d(setpoint.x, setpoint.y, setpoint.psi);
  Eigen::Vector3d nu(state.u, state.v, state.r);
  Eigen::Vector3d eta_dot = J * nu;
  Eigen::Vector3d eta_dot_d(setpoint.u, setpoint.v, setpoint.r);
  Eigen::Vector3d eta_dot_dot_d(setpoint.u_dot, setpoint.v_dot, setpoint.r_dot);

  // INTEGRAL ERROR
  // e_i_dot = sign(err_n)*|err_n|^(q_n/p_n)
  Eigen::Vector3d err(eta_d(0) - eta(0), eta_d(1) - eta(1),
                      angle_dist(eta_d(2), eta(2)));
  Eigen::Vector3d qp(p.x.q / p.x.p, p.y.q / p.y.p, p.psi.q / p.psi.p);
  Eigen::Array3d qp1 = (Eigen::Vector3d::Ones() - qp).array();
  Eigen::Array3d tc(p.x.tc, p.y.tc, p.psi.tc);
  Eigen::Vector3d e_qp = err.cwiseAbs().array().pow(qp.array());
  Eigen::Vector3d e_i_dot = err.cwiseSign().cwiseProduct(e_qp);

  // INITIALIZE ALPHA
  if (!initialized) {
    alpha = (err.array().abs().pow(qp1) / (tc * qp1)).max(1e-6);
    // alpha << 1e-6, 1e-6, 1e-6;
    e_i = -err.cwiseQuotient(alpha);
    initialized = true;
  }

  // SLIDING SURFACE
  // s = e_dot + beta*(e) + alpha*e_I
  e_i = integral_step * (e_i_dot + e_i_dot_last) / 2 + e_i;
  e_i_dot_last = e_i_dot;
  Eigen::Vector3d err_dot = eta_dot_d - eta_dot;
  Eigen::Vector3d s =
      err_dot + beta.cwiseProduct(err) + alpha.cwiseProduct(e_i);

  // ADAPTIVE GAIN
  // K_dot = sqrt(K_a)*sqrt(|s|) - sqrt(K_b)*K^2
  Eigen::Vector3d K_a(p.x.k_alpha, p.y.k_alpha, p.psi.k_alpha);
  Eigen::Vector3d K_b(p.x.k_beta, p.y.k_beta, p.psi.k_beta);
  Eigen::Vector3d s_abs_sqrt = s.cwiseAbs().cwiseSqrt();
  Eigen::Vector3d K_dot = K_a.cwiseSqrt().cwiseProduct(s_abs_sqrt) -
                          K_b.cwiseProduct(K.cwiseProduct(K));
  K = integral_step * (K_dot + K_dot_last) / 2 + K;
  K_dot_last = K_dot;

  // AUXLIARY CONTROL
  Eigen::Vector3d eps(p.x.epsilon, p.y.epsilon, p.psi.epsilon);
  Eigen::Vector3d sign_s = s.cwiseSign();
  Eigen::Vector3d U_aux = -K.cwiseProduct(s_abs_sqrt).cwiseProduct(sign_s) -
                          eps.cwiseProduct(K).cwiseProduct(s.cwiseAbs());

  // DYNAMICS
  DecomposedDyn dyn = model.get_decomposed_dyn(nu);

  // CONTROL SIGNAL
  Eigen::Vector3d U =
      dyn.g_inv *
      (J_inv * (eta_dot_dot_d - J_dot * nu + beta.cwiseProduct(err_dot) +
                alpha.cwiseProduct(e_i_dot) - U_aux) -
       dyn.f);

  // ALLOCATE FORCES
  Azimuth out;
  double Tx = U(0);
  double Ty = U(1);
  double Tz = U(2);

  // Split Tx evenly between front and back thrusters
  double Fx0 = Tx / 2.0;
  double Fx1 = Tx / 2.0;

  // Solve for Fy0 and Fy1 to satisfy both Ty and Tz
  // Since model.lx1 = -model.lx0, Tz = model.lx0 * (Fy0 - Fy1)
  double Fy0 = (Ty + Tz / model.lx0) / 2.0;
  double Fy1 = (Ty - Tz / model.lx0) / 2.0;

  out.force0 = std::hypot(Fx0, Fy0);
  out.force1 = std::hypot(Fx1, Fy1);
  out.ang0 = std::atan2(Fy0, Fx0);
  out.ang1 = std::atan2(Fy1, Fx1);

  // Clamp forces
  if (out.force0 > model.u_max || out.force1 > model.u_max) {
    double max_f = std::max(out.force0, out.force1);
    double scale = model.u_max / max_f;
    out.force0 *= scale;
    out.force1 *= scale;
  }

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
  initialized = false;
}

void AITSMC::reset_integral() {
  e_i = Eigen::Vector3d::Zero();
  e_i_dot_last = Eigen::Vector3d::Zero();
  K = Eigen::Vector3d::Zero();
  K_dot_last = Eigen::Vector3d::Zero();
  initialized = false;
}
