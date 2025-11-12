#ifndef DYNAMICCAST_H
#define DYNAMICCAST_H

namespace sys {

template<typename T, typename U>
bool isa(U *t) {
    return T::classof(t);
}
  
template<typename T, typename U>
T* cast(U *t) {
    assert(isa<T>(t) && "Invalid cast!");
    return (T*) t;
}

template<typename T, typename U>
T* dyn_cast(U *t) {
    if (isa<T>(t)) {
        return cast<T>(t);
    }
    return nullptr;
}

}
#endif // DYNAMICCAST_H