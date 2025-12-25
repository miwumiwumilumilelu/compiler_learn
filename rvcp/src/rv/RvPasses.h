#ifndef RV_RVPASSES_H
#define RV_RVPASSES_H

#include "../CFG-opt/pass.h"
#include "RvAttrs.h"
#include "RvOps.h"
#include "../codegen/Ops.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

namespace sys {
namespace rv {
    
class Lower : public Pass {
private:
    // lambda has Ret operator()(A a) const
    template<typename F,typename Ret, typename A>
    static A helper(Ret (F::*)(A) const);

    template<class F>
    using argument_t = decltype(helper(&F::operator()));

    template<class F>
    void runRewriter(F rewriter) {
        using T = std::remove_pointer_t<argument_t<F>>;
        bool success;
        int total = 0;
        do {
            if (++total > 10000) 
                break;
            
            auto ts = module->findAll<T>();
            success = false;
            for (auto t : ts) {
                success |= rewriter(cast<T>(t));
            }
        } while (success);
    }

public:
    Lower(ModuleOp *module) : Pass(module) {}
    std::string name() override { return "rv-lower"; }
    std::map<std::string, int> stats() override { return {}; }
    void run() override;
};

class RegAlloc : public Pass {
    int spilled = 0;
    int convertedTotal = 0;

    std::map<FuncOp*, std::set<Reg>> usedRegisters;
    std::map<std::string, FuncOp*> fnMap;

    void runImpl(Region *region, bool isLeaf);
    // Create both Prologue and Epilogue of a function.
    void proEpilogue(FuncOp *fn, bool isLeaf);
public:
    RegAlloc(ModuleOp *module) : Pass(module) {}
    std::string name() override { return "rv-regalloc"; }
    std::map<std::string, int> stats() override;
    void run() override;
};

// Dumps the output.
class Dump : public Pass {
    std::string out;

    void dump(std::ostream &os);
public:
    Dump(ModuleOp *module, const std::string &out): Pass(module), out(out) {}

    std::string name() override { return "rv-dump"; };
    std::map<std::string, int> stats() override { return {}; }
    void run() override;
};

}
}

#endif // RV_RVPASSES_H