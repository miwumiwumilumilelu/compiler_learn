# 2025.11.13 前端 —— 语法分析前ASTNode定义

用utils存一些辅助头文件

`src/utils/DynamicCast.h`

## DynamicCast.h

参考LLVM实现

https://stackoverflow.com/questions/6038330/how-is-llvm-isa-implemented

LLVM 广泛使用了一种手工实现的 RTTI 形式，它使用诸如 isa<>、cast<> 和 dyn_cast<> 之类的模板

是其他文件实现继承和多态的基础

```c++
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
```



## Type.h

