#ifndef PASSES_H
#define PASSES_H

#include "pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

#include <set>
#include <map>

namespace sys {

// Converts alloca's to SSA values.
// This must run on flattened CFG, otherwise `break` and `continue` are hard to deal with.
class Mem2Reg : public Pass {
  int count = 0;  // Total converted count
  int missed = 0; // Unconvertible alloca's

  // Maps AllocaOp* to Value (the real value of this alloca).
  using SymbolTable = std::map<Op*, Value>;

  void runImpl(FuncOp *func);
  void fillPhi(BasicBlock *bb, SymbolTable symbols);
  
  // Maps phi to alloca.
  std::map<Op*, Op*> phiFrom;
  std::set<BasicBlock*> visited;
  // Allocas we're going to convert in the pass.
  std::set<Op*> converted;
  DomTree domtree;
public:
  Mem2Reg(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "mem2reg"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

}

#endif