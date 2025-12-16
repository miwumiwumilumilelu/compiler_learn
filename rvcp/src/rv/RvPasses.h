#ifndef RV_RVPASSES_H
#define RV_RVPASSES_H

#include "../opt/pass.h"
#include "RvAttrs.h"
#include "RvOps.h"
#include "../codegen/Ops.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

namespace sys {
namespace rv {
    
class Lower : public Pass {
public:
    Lower(ModuleOp *module) : Pass(module) {}
    std::string name() override { return "rv-lower"; }
    std::map<std::string, int> stats() override { return {}; }
    void run() override;
};

}
}

#endif // RV_RVPASSES_H