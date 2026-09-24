#ifndef USV_DYNAMIC_MODEL_H
#define USV_DYNAMIC_MODEL_H

#include "asv_control/model/dynamic_model.h"

#include <utility>

// Small differential-thrust USV. Parameters and D/C matrices mirror
// scripts/mpc/usv_dynamics.py so the simulator matches the MPC model.
class UsvDynamicModel : public DynamicModel {
public:
  UsvDynamicModel(const Eigen::Vector3d &pose = Eigen::Vector3d::Zero(),
                  const Eigen::Vector3d &vel = Eigen::Vector3d::Zero());

  // Port / starboard thrusts [N]
  State update(double T_port, double T_stbd);
  // Surge force X and yaw moment N -> saturated {port, stbd} thrusts [N]
  static std::pair<double, double> allocate(double X, double N);
  DecomposedDyn get_decomposed_dyn(const Eigen::Vector3d &nu_) override;

  constexpr static double B = 0.41; // Thruster separation [m]
  constexpr static double T_max = 36.5 * 2,
                          T_min = -30 * 2; // Per-thruster limits [N]

  static ModelParams usv_params();
};

#endif // USV_DYNAMIC_MODEL_H
