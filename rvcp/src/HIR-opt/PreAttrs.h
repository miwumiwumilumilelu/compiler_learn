#ifndef PREATTRS_H
#define PREATTRS_H

#include "../codegen/OpBase.h"

#define PREOTLINE __LINE__ + 8388608

namespace sys {

class BaseAttr : public AttrImpl<BaseAttr, PREOTLINE> {
public:
    Op* base;
    BaseAttr(Op* base) : base(base) {}

    std::string toString() override;
    BaseAttr *clone() override { return new BaseAttr(base); }
};

}

#define BASE(op) (op)->get<BaseAttr>()->base

#endif // PREATTRS_H