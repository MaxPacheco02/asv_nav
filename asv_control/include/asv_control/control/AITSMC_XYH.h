// AITSMC with Pose control (x, y, heading)
#ifndef AITSMC_XYH_H
#define AITSMC_XYH_H

#include "AITSMC.h"

struct AITSMC_XYH_Params {
  AITSMCStateParams x, y, psi;
};

class AITSMC_XYH : public AITSMC {
public:
  AITSMC_XYH();
  AITSMC_XYH(const AITSMC_XYH_Params &params);

  Azimuth update(const State &s, const State &setpoint);

private:
  AITSMC_XYH_Params p;
};

#endif
