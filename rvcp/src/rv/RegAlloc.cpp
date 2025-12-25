#include "RvPasses.h"
#include "Regs.h"
#include <unordered_set>

using namespace sys;
using namespace sys::rv;

namespace
{

class SpilledRdAttr : public AttrImpl<SpilledRdAttr, RVLINE + 2097152> {
public:
    bool fp;
    int offset;
    Op* ref;

    SpilledRdAttr(bool fp, int offset, Op* ref) : fp(fp), offset(offset), ref(ref) {}

    std::string toString() override {
        return "<rd-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">";
    }
    SpilledRdAttr* clone() override { return new SpilledRdAttr(fp, offset, ref); }
};

class SpilledRsAttr : public AttrImpl<SpilledRsAttr, RVLINE + 2097152> {
public:
    bool fp;
    int offset;
    Op* ref;

    SpilledRsAttr(bool fp, int offset, Op* ref) : fp(fp), offset(offset), ref(ref) {}

    std::string toString() override {
        return "<rs-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">";
    }
    SpilledRsAttr* clone() override { return new SpilledRsAttr(fp, offset, ref); }
};

class SpilledRs2Attr : public AttrImpl<SpilledRs2Attr, RVLINE + 2097152> {
public:
    bool fp;
    int offset;
    Op* ref;

    SpilledRs2Attr(bool fp, int offset, Op* ref) : fp(fp), offset(offset), ref(ref) {}

    std::string toString() override {
        return "<rs2-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">";  
    }
    SpilledRs2Attr* clone() override { return new SpilledRs2Attr(fp, offset, ref); }
};
    
} // namespace

std::map<std::string, int> RegAlloc::stats() {
    return { 
        {"spilled", spilled}, 
        {"convertedTotal", convertedTotal}
    };
}

// Implemented in OpBase.cpp.
std::string getValueNumber(Value value);

void RegAlloc::runImpl(Region *region, bool isLeaf) {
    
}

void RegAlloc::run() {
    
}



