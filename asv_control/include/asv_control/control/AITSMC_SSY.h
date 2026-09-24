// AITSMC with Velocity control (surge, sway, yaw)
#ifndef AITSMC_SSY_H
#define AITSMC_SSY_H

#include "AITSMC.h"

struct AITSMC_SSY_Params {
  AITSMCStateParams u, v, r;
};

class AITSMC_SSY : public AITSMC {
public:
  AITSMC_SSY();
  AITSMC_SSY(const AITSMC_SSY_Params &params);

  Azimuth update(const State &s, const State &setpoint);
  // Generalized force [X, Y, N] demanded by the control law
  Eigen::Vector3d compute_tau(const State &s, const State &setpoint);
  // Per-axis actuator saturation of the last command (-1, 0, +1). Integration
  // pushing further into saturation is frozen (anti-windup).
  void set_saturation(const Eigen::Vector3d &sat) { saturation = sat; }

private:
  AITSMC_SSY_Params p;
  Eigen::Vector3d saturation{Eigen::Vector3d::Zero()};
};

#endif
