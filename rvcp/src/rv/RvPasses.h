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

}
}

#endif // RV_RVPASSES_H