#include "asv_control/control/PID.h"

PID::PID() = default;

PID::PID(const PIDParams &params) : p(params) {
  kP << p.u.p, p.v.p, p.r.p;
  kI << p.u.i, p.v.i, p.r.i;
  kD << p.u.d, p.v.d, p.r.d;
  kI_max << p.u.i_max, p.v.i_max, p.r.i_max;
}

double PID::normalize_angle(double angle_in) {
  double angle_out = std::fmod(angle_in + M_PI, 2 * M_PI);
  if (angle_out < 0) {
    angle_out += 2 * M_PI;
  }
  return angle_out - M_PI;
}

double PID::angle_dist(double ang1, double ang2) {
  double diff = ang1 - ang2;
  return normalize_angle(diff);
}

Eigen::Matrix3d PID::rotation_matrix(double ang) {
  Eigen::Matrix3d out;
  out << cos(ang), -sin(ang), 0, //
      sin(ang), cos(ang), 0,     //
      0, 0, 1;
  return out;
}

Eigen::Matrix3d PID::rotation_matrix_dot(double ang, double r) {
  Eigen::Matrix3d out;
  out << -sin(ang), -cos(ang), 0, //
      cos(ang), -sin(ang), 0,     //
      0, 0, 0;
  return r * out;
}

Azimuth PID::update(const State &state, const State &setpoint) {
  Eigen::Vector3d eta(state.x, state.y, state.psi);
  Eigen::Vector3d eta_d(setpoint.x, setpoint.y, setpoint.psi);
  Eigen::Vector3d nu(state.u, state.v, state.r);
  Eigen::Vector3d nu_d(setpoint.u, setpoint.v, setpoint.r);
  Eigen::Vector3d nu_dot_d(setpoint.u_dot, setpoint.v_dot, setpoint.r_dot);

  Eigen::Vector3d err = nu_d - nu;

  if (!initialized) {
    err_last = err;
    initialized = true;
  }

  err_i += (err + err_last) / 2 * integral_step;
  err_i = err_i.cwiseMin(kI_max).cwiseMax(-kI_max);
  Eigen::Vector3d err_d = (err - err_last) / integral_step;
  err_last = err;

  // AUXLIARY CONTROL
  Eigen::Vector3d U_aux =
      kP.cwiseProduct(err) + kI.cwiseProduct(err_i) + kD.cwiseProduct(err_d);

  // DYNAMICS
  DecomposedDyn dyn = model.get_decomposed_dyn(nu);

  // CONTROL SIGNAL
  Eigen::Vector3d U = dyn.g_inv * (-dyn.f + U_aux);

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

  // Dead-band: when demanded force is tiny, hold previous angle and zero
  // the magnitude. Avoids azimuth chattering at steady state.
  // Rate-limit: cap angle change to physically realizable slew rate.
  constexpr double F_DEAD = 2222.0; // ~1% of u_max
  constexpr double MAX_SLEW = 0.1;  // rad/step ≈ 10 deg/s @ 100Hz

  auto allocate = [MAX_SLEW](double Fx, double Fy, double &prev_ang) {
    double mag = std::hypot(Fx, Fy);
    if (mag < F_DEAD) {
      return std::make_pair(0.0, prev_ang);
    }
    double desired_ang = std::atan2(Fy, Fx);
    double delta = PID::normalize_angle(desired_ang - prev_ang);
    delta = std::clamp(delta, -MAX_SLEW, MAX_SLEW);
    prev_ang = PID::normalize_angle(prev_ang + delta);
    return std::make_pair(mag, prev_ang);
  };

  std::tie(out.force0, out.ang0) = allocate(Fx0, Fy0, prev_ang0);
  std::tie(out.force1, out.ang1) = allocate(Fx1, Fy1, prev_ang1);

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
            << "U_aux:\n"
            << U_aux.format(fmt) << "\n"
            << "err:\n"
            << err.format(fmt) << "\n"
            << std::endl;

  for (int i = 0; i < 3; i++) {
    debugData[i].e = err(i);
    debugData[i].kP = kP(i);
    debugData[i].kI = kI(i);
    debugData[i].kD = kD(i);
    debugData[i].U = U(i);
  }
  return out;
}
