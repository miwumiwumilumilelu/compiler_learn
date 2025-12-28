#ifndef RVOPS_H
#define RVOPS_H

#include "../codegen/Ops.h"

#define RVOPBASE(ValueTy, Ty) \ 
    class Ty : public OpImpl<Ty, __LINE__ + 520725> { \
    public: \
        explicit Ty(const std::vector<Value> &values): OpImpl(ValueTy, values) { setName("rv."#Ty); } \
        Ty(): OpImpl(ValueTy, {}) { setName("rv."#Ty); } \
        explicit Ty(const std::vector<Attr*> &attrs): OpImpl(ValueTy, {}, attrs) { setName("rv."#Ty); } \
        Ty(const std::vector<Value> &values, const std::vector<Attr*> &attrs): OpImpl(ValueTy, values, attrs) { setName("rv."#Ty); } \
    };

#define RVOPE(Ty) \
    class Ty : public OpImpl<Ty, __LINE__ + 520725> { \
    public: \
        Ty(Value::Type resultTy, const std::vector<Value> &values): OpImpl(resultTy, values) { setName("rv."#Ty); } \
        explicit Ty(Value::Type resultTy): OpImpl(resultTy, {}) { setName("rv."#Ty); } \
        Ty(Value::Type resultTy, const std::vector<Attr*> &attrs): OpImpl(resultTy, {}, attrs) { setName("rv."#Ty); } \
        Ty(Value::Type resultTy, const std::vector<Value> &values, const std::vector<Attr*> &attrs): OpImpl(resultTy, values, attrs) { setName("rv."#Ty); } \
    }

// RVOPBASE is used to mostly RV ops
#define RVOP(Ty) RVOPBASE(Value::i32, Ty)
#define RVOPL(Ty) RVOPBASE(Value::i64, Ty)
#define RVOPF(Ty) RVOPBASE(Value::f32, Ty)

namespace sys {
namespace rv {
// base integer ops
RVOP(LiOp);
RVOPL(LaOp);
RVOPL(AddOp);
RVOP(AddwOp);
RVOP(AddiwOp);
RVOPL(AddiOp);
RVOPL(SubOp);
RVOP(SubwOp);
RVOP(MulwOp);
RVOPL(MulOp);
RVOP(MulhOp);
RVOP(MulhuOp);
RVOP(DivwOp);
RVOPL(DivOp);
RVOP(RemwOp);
RVOPL(RemOp);
// shift ops
RVOP(SlliwOp);
RVOPL(SlliOp);
RVOP (SrliwOp);
RVOPL(SrliOp);
RVOP(SraiwOp);
RVOPL(SraiOp);
RVOP(SllwOp);
RVOPL(SllOp);
RVOP(SrlwOp);
RVOPL(SrlOp);
RVOP(SrawOp);
RVOPL(SraOp);
// logic and bitwise ops
RVOP(AndOp);
RVOP(OrOp);
RVOP(XorOp);
RVOP(AndiOp);
RVOP(OriOp);
RVOP(XoriOp);
// compare ops
RVOP(BneOp);
RVOP(BeqOp);
RVOP(BltOp);
RVOP(BgeOp);
RVOP(BleOp);
RVOP(BgtOp);
RVOP(SeqzOp);
RVOP(SnezOp);
RVOP(SltOp);
RVOP(SltiOp);
// contral flow ops
RVOP(JOp);
RVOP(MvOp);
RVOP(RetOp);
RVOPE(LoadOp);
RVOP(StoreOp);
RVOP(SubSpOp); // sub stack pointer
RVOPE(ReadRegOp);
RVOP(WriteRegOp);
RVOP(CallOp);
RVOP(PlaceHolderOp); // See regalloc; holds a place to denote a register isn't available.
// float ops
RVOPF(FcvtswOp);
RVOP(FcvtwsRtzOp);
RVOPF(FmvwxOp);
RVOPF(FmvdxOp);
RVOPL(FmvxdOp);
RVOP(FldOp);
RVOP(FsdOp);
RVOP(FeqOp);
RVOP(FltOp);
RVOP(FleOp);
RVOPF(FaddOp);
RVOPF(FsubOp);
RVOPF(FmulOp);
RVOPF(FdivOp);
RVOPF(FmvOp);

inline bool hasRd(Op *op) {
    return !(
        isa<StoreOp>(op) ||isa<RetOp>(op) || isa<JOp>(op) ||
        isa<BeqOp>(op) || isa<BneOp>(op) || isa<BltOp>(op) || isa<BgeOp>(op) || isa<BleOp>(op) || isa<BgtOp>(op) ||
        isa<WriteRegOp>(op) || isa<CallOp>(op) || isa<FsdOp>(op)
    );
}

}
}
#undef RVOP
#undef RVOPBASE
#undef RVOPE

#endif // RVOPS_H