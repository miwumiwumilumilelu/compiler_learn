#ifndef TYPECONTEXT_H
#define TYPECONTEXT_H

#include <unordered_set>
#include "Type.h"
#include "../utils/DynamicCast.h"

namespace sys {

// manage memory of Types
class TypeContext {
    struct Hash {
        size_t operator()(Type *ty) const {
            size_t hash = ty->getID();
            
            if(auto arr = dyn_cast<ArrayType>(ty)){
                hash = (hash << 4) + Hash()(arr->base);
                for (auto d : arr->dims) {
                    hash *= (d + 1);
                }
            }

            if(auto fn = dyn_cast<FunctionType>(ty)) {
                hash = (hash << 4) + Hash()(fn->ret);
                for (auto p : fn->params) {
                    hash = (hash << 1) + Hash()(p);
                }
            }

            if(auto ptr = dyn_cast<PointerType>(ty)) {
                hash = (hash << 4) + Hash()(ptr->baseType);
            }

            return hash;
        }
    };

    struct Eq {
        bool operator()(Type *a, Type *b) const {
            if (a->getID() != b->getID()) return false;

            if (auto arrA = dyn_cast<ArrayType>(a)) {
                auto arrB = dyn_cast<ArrayType>(b);
                if (arrA->dims.size() != arrB->dims.size()) return false;
                for (size_t i = 0; i < arrA->dims.size(); ++i) {
                    if (arrA->dims[i] != arrB->dims[i]) return false;
                }
                return Eq()(arrA->base, arrB->base);
            }

            if (auto fnA = dyn_cast<FunctionType>(a)) {
                auto fnB = dyn_cast<FunctionType>(b);
                if (fnA->params.size() != fnB->params.size()) return false;
                for (size_t i = 0; i < fnA->params.size(); ++i) {
                    if (!Eq()(fnA->params[i], fnB->params[i])) return false;
                }
                return Eq()(fnA->ret, fnB->ret);
            }

            if (auto ptrA = dyn_cast<PointerType>(a)) {
                auto ptrB = dyn_cast<PointerType>(b);
                return Eq()(ptrA->baseType, ptrB->baseType);
            }
            return true; // for simple types
        }
    };

    std::unordered_set<Type*, Hash, Eq> content;

public:
    template<class T, class... Args>
    T *create(Args... args) {
        auto ptr = new T(std::forward<Args>(args)...);
        if (auto [it, absent] = content.insert(ptr); !absent) {
            delete ptr;
            return cast<T>(*it);
        }
        return ptr;
    }

    ~TypeContext() {
        for (auto x : content)
        delete x;
    }
};

}

#endif // TYPECONTEXT_H