#include <iostream>
#include <cassert>
#include <vector>

// 包含你需要测试的所有头文件
#include "../utils/DynamicCast.h" // 包含 cast<>
#include "../parse/Type.h"
#include "../parse/TypeContext.h"


int main() {
    sys::TypeContext ctx; // 1. 创建 Type "工厂"

    std::cout << "--- Testing Type System ---" << std::endl;

    // --- 测试 1: 简单类型 ---
    std::cout << "Testing simple types..." << std::endl;
    sys::Type *intTy = ctx.create<sys::IntType>();
    sys::Type *floatTy = ctx.create<sys::FloatType>();
    sys::Type *voidTy = ctx.create<sys::VoidType>();

    std::cout << "IntType toString: " << intTy->toString() << std::endl;
    std::cout << "FloatType toString: " << floatTy->toString() << std::endl;
    
    assert(intTy->toString() == "int");
    assert(floatTy->toString() == "float");
    assert(voidTy->toString() == "void");

    
    std::cout << "Simple types OK." << std::endl;

    // --- 测试 2: 复合类型 (int[3][4]) ---
    std::cout << "Testing composite types..." << std::endl;
    sys::Type *arrTy = ctx.create<sys::ArrayType>(intTy, std::vector<int>{3, 4});

    std::cout << "ArrayType toString: " << arrTy->toString() << std::endl;
    assert(arrTy->toString() == "int[3][4]");
    
    // 【修复】
    // 我们必须使用 cast<> 来告诉编译器 arrTy 是一个 ArrayType
    std::cout << "ArrayType getSize: " << sys::cast<sys::ArrayType>(arrTy)->getSize() << " elements" << std::endl;
    assert(sys::cast<sys::ArrayType>(arrTy)->getSize() == 12); 
    
    std::cout << "ArrayType OK." << std::endl;

    // --- 测试 3: 类型唯一化 (Type Interning) ---
    std::cout << "Testing Type Interning..." << std::endl;
    sys::Type *intTy_2 = ctx.create<sys::IntType>();
    sys::Type *arrTy_2 = ctx.create<sys::ArrayType>(intTy, std::vector<int>{3, 4});

    // 检查指针地址是否相同
    std::cout << "  Address of intTy 1: " << intTy << std::endl;
    std::cout << "  Address of intTy 2: " << intTy_2 << std::endl;
    assert(intTy == intTy_2); // 应该指向同一个对象

    std::cout << "  Address of arrTy 1: " << arrTy << std::endl;
    std::cout << "  Address of arrTy 2: " << arrTy_2 << std::endl;
    assert(arrTy == arrTy_2); // 应该指向同一个对象
    
    std::cout << "Type Interning OK." << std::endl;
    
    // --- 测试 4: 指针和函数类型 ---
    std::cout << "Testing Pointer and Function types..." << std::endl;
    sys::Type *intPtrTy = ctx.create<sys::PointerType>(intTy);
    sys::Type *intPtrPtrTy = ctx.create<sys::PointerType>(intPtrTy);
    
    std::vector<sys::Type*> params = {intTy, floatTy, intPtrTy};
    sys::Type *funcTy = ctx.create<sys::FunctionType>(voidTy, params);
    
    std::cout << "PointerType toString: " << intPtrTy->toString() << std::endl;
    std::cout << "PointerPtrType toString: " << intPtrPtrTy->toString() << std::endl;
    std::cout << "FunctionType toString: " << funcTy->toString() << std::endl;
    
    assert(intPtrTy->toString() == "int*");
    assert(intPtrPtrTy->toString() == "int**");
    assert(funcTy->toString() == "(int,float,int*) -> void"); // 检查你的 interleave 逻辑
    

    std::cout << "Pointer/Function types OK." << std::endl;

    std::cout << "--- All Type Tests Passed ---" << std::endl;
    return 0;
}