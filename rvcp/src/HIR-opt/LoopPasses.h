#ifndef LOOPPASSES_H
#define LOOPPASSES_H

#include "../CFG-opt/pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

namespace sys {

// Raise WhileOp to ForOp.
class RaiseToFor : public Pass {
    int raised = 0;
public:
    RaiseToFor(ModuleOp *module) : Pass(module) {}

    std::string name() override { return "raise-to-for"; }
    std::map<std::string, int> stats() override { return {}; }
    void run() override;
};

// Lower operations back to its original form.
class ForOpLower : public Pass {
public:
    ForOpLower(ModuleOp *module): Pass(module) {}
    
    std::string name() override { return "forop-lower"; }
    std::map<std::string, int> stats() override { return {}; }
    void run() override;
};

}

#endif // LOOPPASSES_H