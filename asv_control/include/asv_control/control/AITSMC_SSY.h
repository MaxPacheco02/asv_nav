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

private:
  AITSMC_SSY_Params p;
};

#endif
