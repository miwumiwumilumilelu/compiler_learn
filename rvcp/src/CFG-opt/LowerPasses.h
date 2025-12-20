#ifndef LOWER_PASSES_H
#define LOWER_PASSES_H

#include "pass.h"

namespace sys {

class FlattenCFG : public Pass {
public:
  FlattenCFG(ModuleOp *module): Pass(module) {}
  
  std::string name() override { return "flatten-cfg"; };
  std::map<std::string, int> stats() override { return {}; }; 
  void run() override;
};

}

#endif LOWER_PASSES_H