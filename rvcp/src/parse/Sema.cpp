#include "Sema.h"
#include "ASTNode.h"
#include "../utils/DynamicCast.h"
#include "Type.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <set>

using namespace sys;

// decay array type to pointer type
PointerType* Sema::decay(ArrayType* arrTy) {
    std::vector<int> dims;
    for (int i = 1; i < arrTy->dims.size(); ++i) {
        dims.push_back(arrTy->dims[i]);
    }
    if (!dims.size()) {  
        return ctx.create<PointerType>(arrTy -> base);
    }
    return ctx.create<PointerType>(ctx.create<ArrayType>(arrTy->base, dims));
}

// raise pointer type to array type
ArrayType* Sema::raise(PointerType* ptr) {
    std::vector<int> dims {1};
    Type *base;
    if (auto pointee = dynamic_cast<ArrayType*>(ptr->baseType)) {
        for (auto x : pointee->dims) {
            dims.push_back(x);
        }
        base = pointee->base;
    }
    else {
        base = ptr->baseType;
    }
    return ctx.create<ArrayType>(base, dims);
}


