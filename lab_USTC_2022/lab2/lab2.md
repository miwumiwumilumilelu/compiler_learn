# _lab2 Light IR C++中间代码生成

## 1. Light IR 预热

### 1.1 实验题

#### 1.1.1 手工编写 IR 文件

LLVM 是一个自由软件项目，它是一种编译器基础设施，以 C++ 写成，包含一系列模块化的编译器组件和工具链，用来开发编译器前端和后端

```
.
├── ...
├── include
│   ├── common
│   └── lightir/*
└── tests
    ├── ...
    └── 2-ir-gen
        └── warmup
            ├── CMakeLists.txt
            ├── c_cases           <- 需要翻译的 c 代码
            ├── calculator        <- 助教编写的计算器示例
            ├── stu_cpp           <- 学生需要编写的 .ll 代码手动生成器
            ├── stu_ll            <- 学生需要手动编写的 .ll 代码
            └── ta_gcd            <- 助教编写的 .ll 代码手动生成器示例
```



```shell
# 可用 clang 生成 C 代码对应的 .ll 文件
$ clang -S -emit-llvm gcd_array.c
# lli 可用来执行 .ll 文件
$ lli gcd_array.ll
# `$?` 的内容是上一条命令所返回的结果，而 `echo $?` 可以将其输出到终端中
$ echo $?
```

实验在 `tests/2-ir-gen/warmup/c_cases/` 目录下提供了四个 C 程序： `assign.c`、 `fun.c`、 `if.c` 和 `while.c`。学生需要在 `test/2-ir-gen/warmup/stu_ll` 目录中，手动使用 LLVM IR 将这四个 C 程序翻译成 IR 代码，得到 `assign_hand.ll`、`func_hand.ll`、`if_handf.ll` 和 `while_hand.ll`，可参考 `clang -S -emit-llvm` 的输出



切换到`c_cases`目录clang生成相关.ll文件,作为standard结果用来验证	

```c
clang -S -emit-llvm assign.c -o ../stu_ll/assign_clang.ll
clang -S -emit-llvm fun.c -o ../stu_ll/fun_clang.ll
clang -S -emit-llvm if.c -o ../stu_ll/if_clang.ll
clang -S -emit-llvm while.c -o ../stu_ll/while_clang.ll    
```

```c
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ ls
assign_clang.ll  assign_hand.ll  fun_clang.ll  fun_hand.ll  if_clang.ll  if_hand.ll  while_clang.ll  while_hand.ll
```



**手动编写 IR**：

**assign.c**:

```c
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/c_cases$ cat assign.c
int main() {
    int a[10];
    a[0] = 10;
    a[1] = a[0] * 2;
    return a[1];
}
```

* `%a0_ptr = getelementptr [10 x i32], [10 x i32]* %a, i32 0, i32 0`
  * <result> = getelementptr <type>, <ptr-type> <ptr>, <index-type> <idx1>, <index-type> <idx2>, ...

​	`[10 x i32]`	源类型：包含 10 个 `i32`的数组

​	`[10 x i32]* %a`	指针：指向数组 `%a`的指针

​	`i32 0`	第一维索引：选择数组本身（`%a[0]`）

​	`i32 0`	第二维索引：选择数组的第 0 个元素（`a[0]`）

* `%a0_val = load i32, i32* %a0_ptr`

  * <result> = load <type>,<ptr-type><ptr>
  * 计算需先访存，内存 → 寄存器，load；计算后写回内存，寄存器 → 内存，store
  
  **`i32`**：        要加载的数据类型
  
  **`i32* %a0_ptr`**：          源指针（指向 `i32`类型的内存地址）
  
  **`%a0_val`**：           存储加载结果的变量。

```c
define i32 @main() {
    %a = alloca [10 x i32]
    %a0_ptr = getelementptr [10 x i32], [10 x i32]* %a, i32 0, i32 0
    store i32 10, i32* %a0_ptr
    %a1_ptr = getelementptr [10 x i32], [10 x i32]* %a, i32 0, i32 1
    %a0_val = load i32, i32* %a0_ptr
    %result = mul i32 %a0_val, 2
    store i32 %result, i32* %a1_ptr
    %ret_val = load i32, i32* %a1_ptr
    ret i32 %ret_val
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli assign_clang.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
20
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ vim assign_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli assign_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
20
```

成功!



**fun.c**:

```c
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/c_cases$ cat fun.c
int callee(int a) { return 2 * a; }
int main() { return callee(110); }
```

```c
define i32 @callee(i32 %a) {
    %result = mul i32 %a, 2
    ret i32 %result
}

define i32 @main() {
    %ret_val = call i32 @callee(i32 110)
    ret i32 %ret_val
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli fun_clang.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
220
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ vim fun_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli fun_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
220
```

成功!



**if.c:**

```c
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/c_cases$ cat if.c
int main() {
    float a = 5.555;
    if (a > 1)
        return 233;
    return 0;
}
```

* ` %cmp = fcmp ogt float %a_val, 1.0`

​	a > 1	true:1 false:0  ——>cmp

* `%ret_val = select i1 %cmp, i32 233, i32 0`

  cmp:1——>233

  ​	0——>0

  即**(a > 1.0) ? 233 : 0**

```c
define i32 @main() {
    %a = alloca float
    store float 5.555e0, float* %a 
    %a_val = load float, float* %a
    %cmp = fcmp ogt float %a_val, 1.0
    %ret_val = select i1 %cmp, i32 233, i32 0
    ret i32 %ret_val
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli if_clang.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
233
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ vim if_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli if_hand.ll 
lli: lli: if_hand.ll:3:17: error: floating point constant invalid for type
    store float 5.555e0, float* %a 
                ^

manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
1
```

**出现浮点数不规范，改用sitofp方法**

```c
define i32 @main() {
    %a = alloca float
    %val = sitofp i32 5555 to float
    store float %val, float* %a 
    %a_val = load float, float* %a
    %cmp = fcmp ogt float %a_val, 1.0
    %ret_val = select i1 %cmp, i32 233, i32 0
    ret i32 %ret_val
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ vim if_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli if_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
233
```

成功!



**while.c:**

```c
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/c_cases$ cat while.c
int main() {
    int a;
    int i;
    a = 10;
    i = 0;
    while (i < 10) {
        i = i + 1;
        a = a + i;
    }
    return a;
}
```

* ` br label %loop_cond`

​	强制跳转到指定的基本块（Basicblock）

​	**`label`**：关键字，表示跳转目标是基本块标签

* **每次访存，计算，紧接着对应一次写回**

```c
define i32 @main() {
    %a_ptr = alloca i32
    %i_ptr = alloca i32
    store i32 10, i32* %a_ptr
    store i32 0, i32* %i_ptr
    br label %loop_cond

loop_cond:
    %i_val = load i32, i32* %i_ptr
    %cmp = icmp slt i32 %i_val, 10
    br i1 %cmp, label %loop_body, label %exit

loop_body:
    %new_i = add i32 %i_val, 1
    store i32 %new_i, i32* %i_ptr
    %a_val = load i32, i32* %a_ptr
    %new_a = add i32 %a_val, %new_i
    store i32 %new_a, i32* %a_ptr
    br label %loop_cond

exit:
    %result = load i32, i32* %a_ptr
    ret i32 %result
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli while_clang.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
65
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ vim while_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ lli while_hand.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup/stu_ll$ echo $?
65
```

成功!



#### 1.1.2 使用 Light IR C++ 库生成 IR 文件

LLVM 项目提供了辅助 IR 生成的 C++ 库，但其类继承关系过于复杂，并且存在很多为了编译性能的额外设计，不利于学生理解 IR 抽象。因此实验依据 LLVM 的设计，为 Light IR 提供了配套简化的 C++ 库

**样例 `gcd_array.c `	`gcd_array_generator.cpp`**

```c
int x[1];
int y[1];

int gcd(int u, int v) {
    if (v == 0)
        return u;
    else
        return gcd(v, u - u / v * v);
}

int funArray(int u[], int v[]) {
    int a;
    int b;
    int temp;
    a = u[0];
    b = v[0];
    if (a < b) {
        temp = a;
        a = b;
        b = temp;
    }
    return gcd(a, b);
}

int main(void) {
    x[0] = 90;
    y[0] = 18;
    return funArray(x, y);
}
```



* 首先需要包含必要的 LLVM 头文件

  ```cpp
  #include "BasicBlock.hpp"   // 基本块
  #include "Constant.hpp"     // 常量
  #include "Function.hpp"     // 函数
  #include "IRBuilder.hpp"    // IR 构建器
  #include "Module.hpp"       // 模块
  #include "Type.hpp"         // 类型
  
  #include <iostream>        // 标准输入输出
  #include <memory>           // 智能指针
  ```

* 定义宏简化常量创建

  ```cpp
  // 整型常量创建宏
  #define CONST_INT(num) \
      ConstantInt::get(num, module)
  
  // 浮点常量创建宏
  #define CONST_FP(num) \
      ConstantFP::get(num, module)
  ```

* 创建 LLVM 模块和 IR 构建器，定义全局数组变量

  ```cpp
  int main() {
      // 创建模块实例
      auto module = new Module();
      
      // 创建 IR 构建器
      auto builder = new IRBuilder(nullptr, module);
      
      // 获取基本类型,使用取出的i32整形与数组长度 1，创建数组类型 [1 x i32]
      Type *Int32Type = module->get_int32_type();
      auto *arrayType = ArrayType::get(Int32Type, 1);
      
      //创建零初始化器
      auto initializer = ConstantZero::get(Int32Type, module);
      
      //创建全局数组 x 和 y
      auto x = GlobalVariable::create("x", module, arrayType, false, initializer);
      auto y = GlobalVariable::create("y", module, arrayType, false, initializer);
  ```

* GCD 函数：实现递归的最大公约数计算函数

  * **函数框架：**创建函数类型（返回类型和参数类型）`gcdFunTy` ——>创建函数实例并命名 `gcdFun` ——>创建入口基本块` bb`
  * **处理函数参数：**为返回值和参数分配栈空间`builder->create_alloca`——>将传入参数存储到分配的栈空间 `builder->create_store`，这里先用vector作为传入参数的载体
  * **判断递归基（v==0）：**

  ```cpp
  // 条件判断
  auto vLoad = builder->create_load(vAlloca);
  auto icmp = builder->create_icmp_eq(vLoad, CONST_INT(0));
  
  // 创建分支
  auto trueBB = BasicBlock::create(module, "trueBB", gcdFun);
  auto falseBB = BasicBlock::create(module, "falseBB", gcdFun);
  auto retBB = BasicBlock::create(module, "", gcdFun);
  builder->create_cond_br(icmp, trueBB, falseBB);
  
  // true 分支 v==0
  builder->set_insert_point(trueBB);
  auto uLoad = builder->create_load(uAlloca);
  builder->create_store(uLoad, retAlloca);
  builder->create_br(retBB);
  
  // false 分支 v!=0
  builder->set_insert_point(falseBB);
  uLoad = builder->create_load(uAlloca);
  vLoad = builder->create_load(vAlloca);
  auto div = builder->create_isdiv(uLoad, vLoad);
  auto mul = builder->create_imul(div, vLoad);
  auto sub = builder->create_isub(uLoad, mul);
  auto call = builder->create_call(gcdFun, {vLoad, sub});
  builder->create_store(call, retAlloca);
  builder->create_br(retBB);
  
  // 返回分支
  builder->set_insert_point(retBB);
  auto retLoad = builder->create_load(retAlloca);
  builder->create_ret(retLoad);
  ```

  

* funArray和main函数同GCD一样



**有以下几点值得注意：**

**内存分配与访问**

```cpp
// 栈上分配
auto alloca = builder->create_alloca(Int32Type);

// 内存存储
builder->create_store(value, ptr);

// 内存加载
auto load = builder->create_load(ptr);
```

**控制流实现**

```cpp
// 条件分支跳转
auto cond = builder->create_icmp_eq(a, b);
builder->create_cond_br(cond, trueBB, falseBB);

// 无条件跳转
builder->create_br(targetBB);
```

**函数调用**

```cpp
// 创建函数调用
auto call = builder->create_call(function, {arg1, arg2});
```

**数组访问**

```cpp
// 获取数组元素指针
auto gep = builder->create_gep(arrayPtr, {index});

// 加载数组元素
auto element = builder->create_load(gep);
```



实验在 `tests/2-ir-gen/warmup/c_cases/` 目录下提供了四个 C 程序。学生需要在 `tests/2-ir-gen/warmup/stu_cpp/` 目录中，参考上面提供的 `gcd_array_generator.cpp` 样例，使用 Light IR C++ 库，编写 `assign_generator.cpp`、`fun_generator.cpp`、`if_generator.cpp` 和 `while_generator.cpp` 四个 cpp 程序。这四个程序运行后应该能够生成 `tests/2-ir-gen/warmup/c_cases/` 目录下四个 C 程序对应的 .ll 文件

**编译：**

```shell
$ cd _lab2/lab2
$ mkdir build
$ cd build
# 使用 cmake 生成 makefile 等文件
$ cmake ..
# 使用 make 进行编译
$ make
```

**运行与测试：**

```shell
# 在 build 目录下操作
$ ./gcd_array_generator > gcd_array_generator.ll
$ lli gcd_array_generator.ll
$ echo $?
```



**assign_generator.cpp:**

```c
int main() {
    int a[10];
    a[0] = 10;
    a[1] = a[0] * 2;
    return a[1];
}
```

```c
#include "BasicBlock.hpp"
#include "Constant.hpp"
#include "Function.hpp"
#include "IRBuilder.hpp"
#include "Module.hpp"
#include "Type.hpp"

#include <iostream>
#include <memory>

#define CONST_INT(num) ConstantInt::get(num, module)
#define CONST_FP(num) ConstantFP::get(num, module)

int main() {
    auto module = new Module();
    auto builder = new IRBuilder(nullptr, module);
    Type *Int32Type = module->get_int32_type();
    auto main = Function::create(FunctionType::get(Int32Type, {}), "main", module);
    auto bb = BasicBlock::create(module, "entry", main);
    builder->set_insert_point(bb);
    auto arrayType = ArrayType::get(Int32Type, 10);
    auto aAlloca = builder->create_alloca(arrayType);
    auto a0GEP = builder->create_gep(aAlloca, {CONST_INT(0), CONST_INT(0)});
    builder->create_store(CONST_INT(10), a0GEP);
    auto a0Load = builder->create_load(a0GEP);
    auto a1Val = builder->create_imul(a0Load, CONST_INT(2));
    auto a1GEP = builder->create_gep(aAlloca, {CONST_INT(0), CONST_INT(1)});
    builder->create_store(a1Val, a1GEP);
    auto retVal = builder->create_load(a1GEP);
    builder->create_ret(retVal);
    std::cout << module->print();
    delete module;
    return 0;
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ ./stu_assign_generator > stu_assign_generator.ll
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ lli stu_assign_generator.ll
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ echo $?
20
```

成功!



**fun_generator.cpp:**

```c
int callee(int a) { return 2 * a; }
int main() { return callee(110); }
```

```cpp
#include "BasicBlock.hpp"
#include "Constant.hpp"
#include "Function.hpp"
#include "IRBuilder.hpp"
#include "Module.hpp"
#include "Type.hpp"

#include <iostream>
#include <memory>

#define CONST_INT(num) ConstantInt::get(num, module)
#define CONST_FP(num) ConstantFP::get(num, module)

int main() {
    auto module = new Module();
    auto builder = new IRBuilder(nullptr, module);
    Type *Int32Type = module->get_int32_type();
    std::vector<Type *> calleeParams = {Int32Type};
    
    auto calleeFun = Function::create(FunctionType::get(Int32Type, calleeParams), "callee", module);
    auto calleeBB = BasicBlock::create(module, "entry", calleeFun);
    builder->set_insert_point(calleeBB);
    auto aAlloca = builder->create_alloca(Int32Type);
    std::vector<Value *> args;
    for (auto &arg : calleeFun->get_args()) {
        args.push_back(&arg);
    }
    builder->create_store(args[0], aAlloca);
    auto aVal = builder->create_load(aAlloca);
    auto result = builder->create_imul(CONST_INT(2), aVal);
    builder->create_ret(result);

    auto mainFun = Function::create(FunctionType::get(Int32Type, {}), "main", module);
    auto mainBB = BasicBlock::create(module, "entry", mainFun);
    builder->set_insert_point(mainBB);
    auto call = builder->create_call(calleeFun, {CONST_INT(110)});
    builder->create_ret(call);

    std::cout << module->print();
    delete module;
    return 0;
}
```

> auto calleeFun = Function::create(FunctionType::get(Int32Type, Int32Type), "callee", module);

不等价于

> std::vector<Type *> calleeParams = {Int32Type};
> auto calleeFun = Function::create(FunctionType::get(Int32Type, calleeParams), "callee", module);

第二个参数错误地直接传递了`Int32Type`（类型对象），而不是参数类型列表

在函数有输入参数的情况下，统一`std::vector<Type*> params = {type1, type2...}`写法，**明确声明参数类型列表**

对应的**取参数**写法:

```cpp
std::vector<Value *> args;
    for (auto &arg : calleeFun->get_args()) {
        args.push_back(&arg);
    }
    builder->create_store(args[0], aAlloca);
```



参考LLVM IR手册中Function的接口:

```c
class Function : public Value {
public:
    // 创建并返回函数，参数依次是待创建函数类型 ty，函数名字 name (不可为空)，函数所属的模块 parent
    static Function *create(FunctionType *ty, const std::string &name, Module *parent);
    // 返回该函数的函数类型
    FunctionType *get_function_type() const;
    // 返回该函数的返回值类型
    Type *get_return_type() const;
    // 将基本块 bb 添加至该函数末端（调用基本块的创建函数时会自动调用此函数来）
    void add_basic_block(BasicBlock *bb);
    // 得到该函数参数数量
    unsigned get_num_of_args() const;
    // 得到该函数基本块数量
    unsigned get_num_basic_blocks() const;
    // 得到该函数所属的 Module
    Module *get_parent() const;
    // 从函数的基本块链表中删除基本块 bb
    void remove(BasicBlock* bb)
    // 返回函数基本块链表
    std::list<BasicBlock *> &get_basic_blocks()
    // 返回函数的参数链表
    std::list<Argument *> &get_args()
    // 给函数中未命名的基本块和指令命名
    void set_instr_name();
};
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ ./stu_fun_generator  > stu_fun_generator.ll
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ lli stu_fun_generator.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ echo $?
220
```

成功!



**if_generator.cpp:**

```c
int main() {
    float a = 5.555;
    if (a > 1)
        return 233;
    return 0;
}
```

```cpp
#include "BasicBlock.hpp"
#include "Constant.hpp"
#include "Function.hpp"
#include "IRBuilder.hpp"
#include "Module.hpp"
#include "Type.hpp"

#include <iostream>
#include <memory>

#define CONST_INT(num) ConstantInt::get(num, module)
#define CONST_FP(num) ConstantFP::get(num, module)

int main(){
    auto module = new Module();
    auto builder = new IRBuilder(nullptr, module);
    Type *Int32Type = module->get_int32_type();
    Type *FloatType = module->get_float_type();
    auto main = Function::create(FunctionType::get(Int32Type, {}), "main", module);
    auto entryBB = BasicBlock::create(module, "entry", main);
    builder->set_insert_point(entryBB);
    auto aAlloca = builder->create_alloca(FloatType);
    builder->create_store(CONST_FP(5.555), aAlloca);
    auto aVal = builder->create_load(aAlloca);
    auto cmp = builder->create_fcmp_gt(aVal, CONST_FP(1.0));
    
    auto thenBB = BasicBlock::create(module, "then", main);
    auto elseBB = BasicBlock::create(module, "else", main);
    builder->create_cond_br(cmp, thenBB, elseBB);

    builder->set_insert_point(thenBB);
    builder->create_ret(CONST_INT(233));

    builder->set_insert_point(elseBB);
    builder->create_ret(CONST_INT(0));

    std::cout << module->print();
    delete module;
    return 0; 
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ ./stu_if_generator > stu_if_generator.ll
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ lli stu_if_generator.ll 
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ echo $?
233
```

成功!



**while_generator.cpp:**

```c
int main() {
    int a;
    int i;
    a = 10;
    i = 0;
    while (i < 10) {
        i = i + 1;
        a = a + i;
    }
    return a;
}
```

```cpp
#include "BasicBlock.hpp"
#include "Constant.hpp"
#include "Function.hpp"
#include "IRBuilder.hpp"
#include "Module.hpp"
#include "Type.hpp"

#include <iostream>
#include <memory>

#define CONST_INT(num) ConstantInt::get(num, module)
#define CONST_FP(num) ConstantFP::get(num, module)

int main() {
    auto module = new Module();
    auto builder = new IRBuilder(nullptr, module);
    Type *Int32Type = module->get_int32_type();
    auto main = Function::create(FunctionType::get(Int32Type, {}), "main", module);
    auto entryBB = BasicBlock::create(module, "entry", main);
    auto loopCondBB = BasicBlock::create(module, "loop_cond", main);
    auto loopBodyBB = BasicBlock::create(module, "loop_body", main);
    auto exitBB = BasicBlock::create(module, "exit", main);
    builder->set_insert_point(entryBB);
    auto a = builder->create_alloca(Int32Type);
    auto i = builder->create_alloca(Int32Type);
    builder->create_store(CONST_INT(10), a);
    builder->create_store(CONST_INT(0), i);
    builder->create_br(loopCondBB);

    builder->set_insert_point(loopCondBB);
    auto iVal = builder->create_load(i);
    auto cmp = builder->create_icmp_lt(iVal, CONST_INT(10));
    builder->create_cond_br(cmp, loopBodyBB, exitBB);

    builder->set_insert_point(loopBodyBB);
    auto iNewVal = builder->create_iadd(iVal, CONST_INT(1));
    builder->create_store(iNewVal, i);
    auto aVal = builder->create_load(a);
    auto aNewVal = builder->create_iadd(aVal, iNewVal);
    builder->create_store(aNewVal, a);
    builder->create_br(loopCondBB);

    builder->set_insert_point(exitBB);
    auto retVal = builder->create_load(a);
    builder->create_ret(retVal);

    std::cout << module->print();
    delete module;
    return 0;
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ ./stu_while_generator > stu_while_generator.ll
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ lli stu_while_generator.ll
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ echo $?
65
```

成功!



### 1.2 思考题

**1.在 [Light IR 简介](https://ustc-compiler-principles.github.io/2023/common/LightIR/)里，你已经了解了 IR 代码的基本结构，请尝试编写一个有全局变量的 cminus 程序，并用 `clang` 编译生成中间代码，解释全局变量在其中的位置。**

```c
int i;
float f = 3.14;

int main() {
    i = 10;
    return 0;
}
```

```shell
manbin@compile:~/2023_warm_up_b/_lab2/lab2/build$ cd ../tests/2-ir-gen/warmup/
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup$ clang -S -emit-llvm test.c -o test.ll
manbin@compile:~/2023_warm_up_b/_lab2/lab2/tests/2-ir-gen/warmup$ cat test.ll
```

```ABAP
; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@f = dso_local global float 0x40091EB860000000, align 4
@i = dso_local global i32 0, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 10, i32* @i, align 4
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
```

* 全局变量 `@f`和 `@i`位于 IR 文件的**最外层作用域**，与函数 `@main`处于同一层级

> Module (test.c) 
>
> ├── Global Variables
>
>  │   	├── @f = global float 3.14
>
>  │   	└── @i = global i32 0
>
> ├── Function (@main)
>
> └── Metadata (!llvm.module.flags, !llvm.ident)

* 全局变量在程序运行时位于**静态数据区**（编译时分配固定地址），与栈（局部变量）或堆（动态分配）无关

​	通过符号名（如 `@i`）直接寻址，无需动态内存分配



**2.Light IR 中基本类型 label 在 Light IR C++ 库中是如何用类表示的？**

`label`类型（基本块标签），通过 `llvm::BasicBlock`类间接表示，由 `llvm::BasicBlock`类封装，每个 `BasicBlock`对象对应一个带标签的代码块；其标记基本块入口，同时向分支指令（如 `br`）提供跳转目标

内存中通过 `BasicBlock*`指针操作标签，而非字符串（如分支指令 `br`的目标参数为 `BasicBlock*`类型）

`label`无大小、无数值，仅作为位置标识符使用，无法存储或参与算术运算；作为上下文相关的隐式类型，由控制流指令动态管理

`llvm`将标签视为基本块的逻辑入口标识符，而非传统的数据类型

**例子:**

`auto entryBB = BasicBlock::create(module, "entry", main);`

entry即为标签，是basicblock的名字



**3.Light IR C++ 库中 `Module` 类中对基本类型与组合类型存储的方式是一样的吗？请尝试解释组合类型使用其存储方式的原因。**

**Module接口:**

```c
class Module
{
public:
    // 将函数 f 添加到该模块的函数链表上
    // 在函数被创建的时候会自动调用此方法
    void add_function(Function *f);
    // 将全局变量 g 添加到该模块的全局变量链表上
    // 在全局变量被创建的时候会自动调用此方法
    void add_global_variable(GlobalVariable* g);
    // 获取全局变量列表
    std::list<GlobalVariable *> get_global_variable();
    // 获得（创建）自定义的 Pointer 类型
    PointerType *get_pointer_type(Type *contained);
    // 获得（创建）自定义的 Array 类型
    ArrayType *get_array_type(Type *contained, unsigned num_elements);
    // 获得基本类型 int32
    IntegerType *get_int32_type();
    // 其他基本类型类似...
};
```

* **基本类型存储方式：单例模式**

1. **连续存储**

   基本类型（如 `i32`、`float`）在内存中占用固定大小的连续空间，例如：

   - `i32`占用 4 字节连续内存
   - `float`占用 4 字节  存储时无需额外元数据，仅按类型大小分配空间

2. **无填充对齐**

   基本类型通常按**自然对齐**（Natural Alignment）存储：

   - 例如 4 字节的 `i32`起始地址需为 4 的倍数对齐由硬件要求决定，确保高效内存访问

3. **直接寻址**

   通过类型指针（如 `i32*`）可直接访问值，无需解析内部结构

4. **每个基本类型在模块中仅存储一份单例对象**

   如：`IntegerType *get_int32_type();`

​	以上**单例模式**可保证**类型唯一性**并减少内存开销

* **组合类型的存储方式：动态创建与缓存**

  ```c
  PointerType *get_pointer_type(Type *contained);
  ArrayType *get_array_type(Type *contained, unsigned num_elements);
  ```

  这些方法会检查是否已存在相同结构的类型：

  - 若存在 → 返回缓存的对象指针
  - 若不存在 → 动态创建新类型并加入缓存

  **内存管理**：组合类型对象存储在 `Module`内部的类型缓存表中（如 `std::map<TypeKey, Type*>`），键值由类型特征决定（如指针指向的元素类型 + 数组元素数量）

__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## 2. 访问者模式

Visitor Pattern（访问者模式）是一种在 LLVM 项目源码中被广泛使用的设计模式。在本实验中，指的是**语法树**类有一个方法接受**访问者**，将自身引用传入**访问者**，而**访问者**类中集成了根据语法树节点内容生成 IR 的规则

### 2.1 实验题

在 `tests/2-ir-gen/warmup/calculator` 目录下提供了一个接受算术表达式，利用访问者模式，产生计算算数表达式的中间代码的程序，其中 `calc_ast.hpp` 定义了语法树的不同节点类型，`calc_builder.cpp` 实现了访问不同语法树节点 `visit` 函数。**阅读这两个文件和目录下的其它相关代码**，理解语法树是如何通过访问者模式被遍历的，并回答相应思考题

**编译**

```shell
$ cd ~/2023_warm_up_b/_lab2/lab2/build
# 使用 cmake 生成 makefile 等文件
$ cmake ..
# 使用 make 进行编译
$ make
```

**运行与测试**

```shell
# 在 build 目录下操作
$ ./calc
Input an arithmatic expression (press Ctrl+D in a new line after you finish the expression):
4 * (8 + 4 - 1) / 2
result and result.ll have been generated.
$ ./result
22
```



#### 2.1.1 **calc_ast.hpp**

**(语法树不同节点类型)**

**`virtual void accept(CalcASTVisitor &) override final;`**声明一个不可再重写的虚函数，并强制要求其必须覆盖基类中的同名虚函数。

```c
class CalcASTVisitor {
  public:
    virtual void visit(CalcASTInput &) = 0;
    virtual void visit(CalcASTNum &) = 0;
    virtual void visit(CalcASTExpression &) = 0;
    virtual void visit(CalcASTTerm &) = 0;
};
```

虚函数，支持运行时多态（派生类可通过重写改变行为）

**在访问者模式中，用于实现节点接受访问者的入口方法，且不允许子类修改其行为**

**`final`**：禁止后续派生类重写此函数，终结继承链中的重写行为



**`virtual void accept(CalcASTVisitor &) override;`**声明一个重写基类虚函数的方法，强调其覆盖意图



**`std::shared_ptr<CalcASTExpression> expression;`**声明一个**智能指针**，指向`CalcASTExpression`类型的对象

**所有权管理**：`std::shared_ptr`自动释放内存，避免泄漏（通过引用计数实现）



**高优先级运算符下沉到更深层级**：在AST中，优先级高的运算符会被解析为更深层级的子树。例如：

- 表达式 `1 + 2 * 3`的AST中，乘法（`*`）优先级高于加法（`+`），因此 `2 * 3`会作为子树的右节点嵌套在加法节点下，形成层级更深的子树

> (+)
>   ├── 1 
>   └── (*) 
>         ├── 2 
>         └── 3





在抽象语法树（AST）的设计中，`CalcASTExpression`（表达式节点）和`CalcASTTerm`（项节点）采用左右两侧结构（如 `expression + op + term`或 `term + op + factor`）

* `CalcASTInput *get_root() { return root.get(); }`

  根节点，包含一个表达式（expression）

  ```c
  struct CalcASTInput : CalcASTNode {
      virtual void accept(CalcASTVisitor &) override final;
      std::shared_ptr<CalcASTExpression> expression;
  };
  ```

  `accept()`实现访问者模式，允许外部遍历器访问自身

  通过 `expression`智能指针连接表达式子树

* ```c
  struct CalcASTExpression : CalcASTFactor {
      void accept(CalcASTVisitor &) override final;
      std::shared_ptr<CalcASTExpression> expression; // 左递归表达式
      AddOp op; // 操作符(+ / -)
      std::shared_ptr<CalcASTTerm> term; // 右侧项
  };
  ```

* ```c
  struct CalcASTTerm : CalcASTNode {
      virtual void accept(CalcASTVisitor &) override final;
      std::shared_ptr<CalcASTTerm> term; // 左递归项
      MulOp op; // 操作符(x / \)
      std::shared_ptr<CalcASTFactor> factor; // 右侧因子
  };
  ```

* ```c
  struct CalcASTFactor : CalcASTNode {
      virtual void accept(CalcASTVisitor &) override;
  };
  ```

  数字 or 括号表达式——表示**不可再分的计算单元**

  - **派生类**
    - **`CalcASTNum`**：叶子节点，存储整数值
    - **`CalcASTExpression`**：括号表达式（如 `(1+2)`退化为一个因子）

* ```c
  struct CalcASTNum : CalcASTFactor {
      virtual void accept(CalcASTVisitor &) override final;
      int val;
  };
  ```

  AST的叶子节点，无子节点



**(1 + 2) * 3:**

> CalcASTInput
> └── CalcASTExpression (OP_MUL)
>     ├── CalcASTTerm (左操作数: 括号表达式)
>     │   └── CalcASTFactor
>     │       └── CalcASTExpression (OP_PLUS)
>     │           ├── CalcASTTerm (左项: 1)
>     │           │   └── CalcASTFactor
>     │           │       └── CalcASTNum(val=1)
>     │           ├── OP_PLUS
>     │           └── CalcASTTerm (右项: 2)
>     │               └── CalcASTFactor
>     │                   └── CalcASTNum(val=2)
>     ├── OP_MUL
>     └── CalcASTFactor (右因子: 3)
>         └── CalcASTNum(val=3)



#### 2.1.2 calc_builder.cpp

- **访问者基类**：`CalcASTVisitor`（在`calc_ast.hpp`中声明）
  - 为每种AST节点类型定义`visit`方法（如`visit(CalcASTInput&)`、`visit(CalcASTExpression&)`等）
- **具体访问者**：`CalcBuilder`（在`calc_builder.cpp`中实现）
  - 继承`CalcASTVisitor`，实现所有`visit`方法。
  - **核心作用**：遍历AST并生成LLVM IR指令





- **元素基类**：`CalcASTNode`（在`calc_ast.hpp`中声明）
  - 定义`accept`方法：`virtual void accept(CalcASTVisitor&) = 0`
- **具体元素**：各类AST节点（如`CalcASTExpression`、`CalcASTTerm`）
  - 实现`accept`方法：调用访问者的`visit`方法并传入自身引用



创建llvm模块和IR构建器

`module = std::unique_ptr<Module>(new Module()); `

`builder = std::make_unique<IRBuilder>(nullptr, module.get());`

然后设置基本块bb



```c
void CalcBuilder::visit(CalcASTInput &node) { 
    node.expression->accept(*this); // 启动表达式子树的遍历
}
```

直接访问表达式子节点，启动整个AST的遍历



```c
void visit(CalcASTExpression &node) {
    if (node.expression == nullptr) { // 单操作数（如直接数字或项）
        node.term->accept(*this);    // 直接访问项节点
    } else {                          // 二元运算（如 a + b）
        node.expression->accept(*this); // 递归访问左侧表达式
        auto l_val = val;              // 保存左操作数结果
        node.term->accept(*this);      // 访问右侧项
        auto r_val = val;              // 保存右操作数结果
        switch (node.op) {            // 根据运算符生成IR指令
            case OP_PLUS:  val = builder->create_iadd(l_val, r_val); break;
            case OP_MINUS: val = builder->create_isub(l_val, r_val); break;
        }
    }
}
```

**递归下降**：先处理左子树，再处理右子树，符合表达式求值顺序

**优先级实现**：表达式节点仅处理加减法，高优先级的乘除法由`CalcASTTerm`处理，确保`3+2 * 1`中乘法优先生成IR



```c
void visit(CalcASTTerm &node) {
    if (node.term == nullptr) {        // 单操作数（如因子）
        node.factor->accept(*this); 
    } else {                          // 二元运算（如 a * b）
        node.term->accept(*this);      // 递归访问左侧项
        auto l_val = val; 
        node.factor->accept(*this);    // 访问右侧因子
        auto r_val = val;
        switch (node.op) {            // 生成乘除指令
            case OP_MUL: val = builder->create_imul(l_val, r_val); break;
            case OP_DIV: val = builder->create_isdiv(l_val, r_val); break;
        }
    }
}
```



```c
void visit(CalcASTNum &node) {
    val = ConstantInt::get(node.val, module.get()); // 生成整数常量IR
}
```

叶子节点



**举例:**

**(1+2) * 3**

整个表达式本质是乘法运算，无需外层 `CalcASTExpression`包装

即无需统一管理加减（低优先级）和乘除（高优先级）的运算顺序

> CalcASTInput
>   └── CalcASTTerm (OP_MUL)
>        ├── CalcASTFactor (括号表达式)
>        │    └── CalcASTExpression (OP_PLUS)
>        │         ├── CalcASTTerm → CalcASTNum(1)
>        │         └── CalcASTTerm → CalcASTNum(2)
>        └── CalcASTFactor → CalcASTNum(3)

1. **IR生成步骤**：

   - 访问`CalcASTTerm`（乘法）：
     1. 递归访问左因子（括号表达式）→ 进入`CalcASTExpression`（加法）
     2. 加法节点：
        - 访问左项（`CalcASTNum(1)`）→ `val = 常量1`
        - 访问右项（`CalcASTNum(2)`）→ `val = 常量2`
        - 生成`iadd`指令 → `val = %add = iadd 1, 2`
     3. 访问右因子（`CalcASTNum(3)`）→ `val = 常量3`
     4. 生成`imul`指令 → `%mul = imul %add, 3`

2. **最终IR**：

   ```cpp
   %add = iadd i32 1, i32 2
   %mul = imul i32 %add, i32 3
   ```



**1 + 2 - 3**

> CalcASTInput
> └── CalcASTExpression (OP_MINUS)
>     ├── CalcASTExpression (OP_PLUS)  // 左子树：1+2
>     │   ├── CalcASTTerm → CalcASTNum(1)
>     │   └── CalcASTTerm → CalcASTNum(2)
>     └── CalcASTTerm → CalcASTNum(3)  // 右操作数

1. **IR 生成步骤**：

- **访问根节点 `CalcASTExpression`（减法 `OP_MINUS`）**：

  1. **处理左子树（加法 `OP_PLUS`）**：

     - 递归访问左操作数（`CalcASTNum(1)`）→ `val = ConstantInt::get(1)`（生成常量 `1`）

     - 递归访问右操作数（`CalcASTNum(2)`）→ `val = ConstantInt::get(2)`（生成常量 `2`）

       生成加法指令：

       `%add_tmp = iadd i32 1, i32 2`（将结果暂存为 `%add_tmp`）

  2. **处理右子树（常数 `3`）**：

     - 访问 `CalcASTNum(3)`→ `val = ConstantInt::get(3)`（生成常量 `3`）

  3. **生成根节点减法指令**：

     - 将左子树结果（`%add_tmp`）与右子树常量（`3`）结合，生成：

       `%sub_tmp = isub i32 %add_tmp, i32 3`（计算 `%add_tmp - 3`）

2. **最终 IR**：

```cpp
%add_tmp = iadd i32 1, i32 2   ; 1 + 2 → %add_tmp
%sub_tmp = isub i32 %add_tmp, i32 3   ; %add_tmp - 3 → %sub_tmp
```



### 2.2 思考题

1. 分析 `calc` 程序在输入为 `4 * (8 + 4 - 1) / 2` 时的行为：

   a.请画出该表达式对应的抽象语法树（使用 `calc_ast.hpp` 定义的语法树节点来表示，并给出节点成员存储的值），并给节点使用数字编号。

   > CalcASTInput [1] 
   > └── CalcASTExpression [2] 
   >     └── CalcASTTerm [3] (op = OP_DIV) 
   >         ├── CalcASTTerm [4] (op = OP_MUL) 
   >         │   ├── CalcASTFactor [5] 
   >         │   │   └── CalcASTNum [6] (val = 4) 
   >         │   └── CalcASTFactor [7] 
   >         │       └── CalcASTExpression [8] (op = OP_MINUS)  // 括号内整体表达式
   >         │           ├── CalcASTExpression [9] (op = OP_PLUS)  // 左操作数：8+4
   >         │           │   ├── CalcASTTerm [10] 
   >         │           │   │   └── CalcASTFactor [11] 
   >         │           │   │       └── CalcASTNum [12] (val = 8) 
   >         │           │   └── CalcASTTerm [13] 
   >         │           │       └── CalcASTFactor [14] 
   >         │           │           └── CalcASTNum [15] (val = 4) 
   >         │           └── CalcASTTerm [16]  // 右操作数：1
   >
   > ​        │               └── CalcASTFactor [17] 
   > ​        │                   └── CalcASTNum [18] (val = 1)
   > ​        └── CalcASTFactor [19]
   > ​            └── CalcASTNum [20] (val = 2)

   **遵循深层优先级高**

   **最高优先级：括号内计算**

   `7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18`

   **次高优先级：乘法**

   `4 → 5 → 6`

   `[7]的结果`

   **最低优先级：除法**（由于**结合性方向**，所以除法在这个例子中低于乘法）！！！

   `[3]整合结果`

   `19 → 20`

   

   b.请给出示例代码在用访问者模式遍历该语法树时，访问者到达语法树节点的顺序。序列请按如下格式指明（序号为问题 1.a 中的编号）：3->2->5->1->1

   `1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20`

1. **入口与根节点**
   - 访问 `CalcASTInput [1]`→ 进入其子节点 `CalcASTExpression [2]`
   - 访问 `CalcASTExpression [2]`→ 调用其子项 `CalcASTTerm [3]`（除法操作）
2. **除法左子树（乘法）**
   - 访问 `CalcASTTerm [3]`（除法）→ 优先访问左操作数 `CalcASTTerm [4]`（乘法操作）
     - 访问乘法左因子 `CalcASTFactor [5]`→ 进入其子节点 `CalcASTNum [6]`（值 `4`）
     - 访问乘法右因子 `CalcASTFactor [7]`（括号表达式）→ 进入其子节点 `CalcASTExpression [8]`
3. **括号内表达式（加减法）**
   - 访问 `CalcASTExpression [8]`（括号内整体表达式）→ 先访问左子树 `CalcASTExpression [9]`（加法）
     - 访问加法左项 `CalcASTTerm [10]`→ 因子 `CalcASTFactor [11]`→ 数字 `CalcASTNum [12]`（值 `8`）
     - 访问加法右项 `CalcASTTerm [13]`→ 因子 `CalcASTFactor [14]`→ 数字 `CalcASTNum [15]`（值 `4`）
   - 访问 `CalcASTExpression [8]`的右子树 `CalcASTTerm [16]`（减法右操作数）
     - 访问 `CalcASTTerm [16]`→ 因子 `CalcASTFactor [17]`→ 数字 `CalcASTNum [18]`（值 `1`）
4. **除法右子树（数字）**
   - 回溯到除法节点 `[3]`→ 访问右操作数 `CalcASTFactor [19]`→ 数字 `CalcASTNum [20]`（值 `2`）



## 3. IR 自动化生成 (实验)

在实验框架下，利用访问者模式遍历抽象语法树，调用 Light IR C++ 库，实现 IR 自动化生成

实验框架实现了 Lab1 生成的分析树到 C++ 上的抽象语法树的转换。可以使用[访问者模式]来实现对抽象语法树的遍历，**[ast.hpp]文件中包含抽象语法树节点定义**

### 3.1 实验框架介绍

实验框架实现了 Lab1 生成的分析树到 C++ 上的抽象语法树的转换。可以使用[访问者模式]来实现对抽象语法树的遍历，**[ast.hpp]文件中包含抽象语法树节点定义**

`CminusfBuilder` 类定义在 [cminusf_builder.hpp]文件中，`CminusfBuilder` 类中定义了对抽象语法树不同语法节点的 `visit` 函数，实验已给出了一些语法树节点的访问规则，其余的需要学生补充

**scope符号表：**

`Scope` 类的作用是管理编译器或程序中的作用域（Scope），用于存储和查找变量或符号的定义。它通过维护一个嵌套的作用域结构（类似栈的结构）来实现符号的作用域管理。以下是其主要功能和用途(看注释！看注释！看注释!）

```cpp
class Scope {
  public:
    // enter a new scope
    // enter()：进入一个新的作用域（创建一个新的作用域层）
    void enter() { inner.emplace_back(); }

    // exit a scope
    // 退出当前作用域（移除最近的作用域层）
    void exit() { inner.pop_back(); }
	// 判断当前是否处于全局作用域（只有一个作用域层时为全局作用域）
    bool in_global() { return inner.size() == 1; }

    // push a name to scope
    // return true if successful
    // return false if this name already exits
    // 向当前作用域添加一个符号（变量或函数）。如果符号已存在，则返回 false，否则返回 true
    bool push(const std::string& name, Value *val) {
        auto result = inner[inner.size() - 1].insert({name, val});
        return result.second;
    }
	// 从最近的作用域开始查找符号，直到找到为止。如果找不到符号，会触发断言错误
    Value *find(const std::string& name) {
        for (auto s = inner.rbegin(); s != inner.rend(); s++) {
            auto iter = s->find(name);
            if (iter != s->end()) {
                return iter->second;
            }
        }

        // Name not found: handled here?
        assert(false && "Name not found in scope");

        return nullptr;
    }
	// 来存储作用域层次结构：
    // std::vector：表示作用域的栈结构，每个作用域层是一个std::map
    // std::map<std::string, Value *>：表示符号表，存储符号名称和对应的值（Value *) 
  private:
    std::vector<std::map<std::string, Value *>> inner;
};
```



在 `CminusfBuilder` 构造函数函数中，下列代码片段是对 [Cminusf 语义]中的 4 个预定义函数进行声明并加入全局符号表中，在生成 IR 时可从符号表中查找。我们的测试样例会使用这些函数，从而实现 IO

```c
scope.enter();
scope.push("input", input_fun);
scope.push("output", output_fun);
scope.push("outputFloat", output_float_fun);
scope.push("neg_idx_except", neg_idx_except_fun);
```



`CminusfBuilder` 类使用成员 `context` 存储翻译时状态，下列代码片段是 `context` 的定义，学生需要为该结构体添加更多域来存储翻译时的状态

```c
struct {
    // function that is being built
    Function *func = nullptr;
} context;
```



### 3.2 实验内容和流程理解

阅读lab1的 [Cminusf 语义]，并根据语义补全 `include/cminusfc/cminusf_builder.hpp` 与 `src/cminusfc/cminusf_builder.cpp` 文件，实现 IR 自动产生的算法，使得它能正确编译任何合法的 Cminusf 程序，生成符合 [Cminusf 语义]的 IR

补充内容在实验框架中有提到

1. 请比较通过 cminusfc 产生的 IR 和通过 clang 产生的 IR 来找出可能的问题或发现新的思路。
2. 使用 GDB 进行调试来检查错误的原因。
3. 我们为 `Function`、`Type` 等类都实现了 `print` 接口，可以使用我们提供的 [logging 工具]进行打印调试。

```
.
├── CMakeLists.txt
├── include                             <- 实验所需的头文件
│   ├── ...
|   ├── cminusfc
|   |    └── cminusf_builder.hpp        <- 该阶段需要修改的文件
│   ├── lightir/*
│   └── common
│        ├── ast.hpp
│        ├── logging.hpp
│        └── syntax_tree.h
├── src
│   ├── ...
│   └── cminusfc
│       ├── cminusfc.cpp                <- cminusfc 的主程序文件
│       └── cminusf_builder.cpp         <- 该阶段需要修改的文件
└── tests
    ├── ...
    └── 2-ir-gen
        └── autogen
            ├── testcases                <- 助教提供的测试样例
            ├── answers                  <- 助教提供的测试样例
            └── eval_lab2.py             <- 助教提供的测试脚本
```

**编译**

```shell
$ cd _lab2/lab2/build
# 使用 cmake 生成 makefile 等文件
$ cmake ..
# 使用 make 进行编译
$ make
# 安装以链接 libcminus_io.a
$ sudo make install
```

如果构建成功，会在 `build` 文件夹下找到 cminusfc 可执行文件，它能将 cminus 文件输出为 IR 文件，编译成二进制可执行文件

**运行**

在 `tests/testcases_general` 文件夹中有一些通用案例。当需要对 `.cminus` 单个文件测试时，可以这样使用：

```shell
# 1.生成 IR 文件
# 假设 cminusfc 的路径在你的 $PATH 中，并且你现在在 test.cminus 文件所在目录中
$ cminusfc test.cminus -emit-llvm
#此时会在同目录下生成同名的 .ll 文件，在这里即为 test.ll

# or

# 2.生成可执行文件
# 上面生成的 .ll 文件用于阅读，如果需要运行，需要调用 clang 编译链接生成二进制文件 test
$ clang -O0 -w -no-pie test.ll -o test -lcminus_io
```

**测试**

自动测试脚本和所有测试样例都是公开的，它在 `tests/2-ir-gen/autogen` 目录下，使用方法如下:

```shell
# 在 tests/2-ir-gen/autogen 目录下运行：
$ python3 ./eval_lab2.py
$ cat eval_result
```

测试结果会输出到 `tests/2-ir-gen/autogen/eval_result`



**大概流程梳理访问者模式，以ASTProgram为例：**

```cpp
// 在 ast.hpp
struct ASTProgram : ASTNode {
    virtual Value *accept(ASTVisitor &) override final;
    virtual ~ASTProgram() = default;
    std::vector<std::shared_ptr<ASTDeclaration>> declarations;
};
```

```cpp
// 在 ast.cpp
Value* ASTProgram::accept(ASTVisitor &visitor) {
    return visitor.visit(*this); // 调用访问者的 visit(ASTProgram&) 方法
}
```

- **参数**：接收一个 `ASTVisitor`对象（如 `CminusfBuilder`）,类似《规则选择书》
- **行为**：将自身（`*this`）传递给访问者的 `visit`方法，触发具体的处理逻辑



由以上得知`ASTProgram`继承自 `ASTNode` ，可简化为：

```cpp
struct ASTProgram {
    std::vector<std::shared_ptr<ASTDeclaration>> declarations;
    Value* accept(ASTVisitor& v) { return v.visit(*this); }
};
```

**`ASTProgram`**就像一个大仓库，里面存放着所有快递包裹（变量声明、函数声明等)；当有人来取件时，管理员（`accept`）会说："请按照『程序包裹』的标准流程处理我"



```c++
// ast.hpp
class ASTVisitor {
  public:
    virtual Value *visit(ASTProgram &) = 0;
    virtual Value *visit(ASTNum &) = 0;
    virtual Value *visit(ASTVarDeclaration &) = 0;
    virtual Value *visit(ASTFunDeclaration &) = 0;
    virtual Value *visit(ASTParam &) = 0;
    virtual Value *visit(ASTCompoundStmt &) = 0;
    virtual Value *visit(ASTExpressionStmt &) = 0;
    virtual Value *visit(ASTSelectionStmt &) = 0;
    virtual Value *visit(ASTIterationStmt &) = 0;
    virtual Value *visit(ASTReturnStmt &) = 0;
    virtual Value *visit(ASTAssignExpression &) = 0;
    virtual Value *visit(ASTSimpleExpression &) = 0;
    virtual Value *visit(ASTAdditiveExpression &) = 0;
    virtual Value *visit(ASTVar &) = 0;
    virtual Value *visit(ASTTerm &) = 0;
    virtual Value *visit(ASTCall &) = 0;
};
```

**`ASTVisitor`**就像快递公司的《操作手册》，规定如何拆不同类型的包裹



```cpp
// cminusf_builder.hpp
class CminusfBuilder : public ASTVisitor {
	public:
		......
	private:
		......
		virtual Value *visit(ASTProgram &) override final;
		......
}
```

```cpp
// cminusf_builder.cpp
Value* CminusfBuilder::visit(ASTProgram &node) {
    VOID_T = module->get_void_type();
    INT1_T = module->get_int1_type();
    INT32_T = module->get_int32_type();
    INT32PTR_T = module->get_int32_ptr_type();
    FLOAT_T = module->get_float_type();
    FLOATPTR_T = module->get_float_ptr_type();

    Value *ret_val = nullptr;
    for (auto &decl : node.declarations) {
        ret_val = decl->accept(*this);
    }
    return ret_val;
}
```

先进入ASTProgram的accept——>再进入CminusfBuilder的visit(ASTProgram&)，以下程序

```cpp
    Value *ret_val = nullptr;
    for (auto &decl : node.declarations) {
        ret_val = decl->accept(*this);
    }
    return ret_val;
```

遍历所有声明，递归处理

**`decl->accept(*this)`**：每个声明节点（如 `ASTVarDeclaration`）调用自己的 `accept`方法

而

**`CminusfBuilder`（快递员小哥）**是真正干活的快递员，按照公司规范实际操作:

1. 拿到`ASTProgram`包裹后，先看《手册》找到对应规则
2. 打开包裹发现里面还有小包裹（变量/函数声明）
3. 对每个小包裹说："请按照你自己的类型告诉我该怎么拆你"（调用`accept`）

**类比快递系统：**

|      组件      |     快递比喻     |   关键方法    |      在哪实现       |
| :------------: | :--------------: | :-----------: | :-----------------: |
|    ASTNode     |     快递包裹     |    accept     | 每个节点类单独实现  |
|   ASTVisitor   | 《快递处理手册》 | visit（声明） |     接口不实现      |
| CminusfBuilder |   真正的快递员   | visit（实现） | cminusf_builder.cpp |



以上内容皆是编译器后端 **IR生成阶段**：核心任务是将 **AST转换为LLVM IR**

|       组件       |                          专业职责                           |
| :--------------: | :---------------------------------------------------------: |
|   `ASTVisitor`   |    定义IR生成的抽象接口（纯虚函数），连接访问者和AST节点    |
| `CminusfBuilder` |           具体访问者，实现AST到LLVM IR的转换规则            |
|  `IRBuilder<>`   | LLVM提供的指令生成工具（创建`alloca`/`store`/`load`等指令） |
|     `Module`     |          LLVM顶层容器，管理全局变量、函数等IR实体           |

 1.**双重分发（Double Dispatch）**实现 **AST节点类型 × 访问者类型** 的双重动态绑定

```c++
// AST节点侧
Value* ASTVarDeclaration::accept(ASTVisitor &v) {
    return v.visit(*this); // 第一次分发：根据AST节点类型
}

// 访问者侧
Value* CminusfBuilder::visit(ASTVarDeclaration &node) {
    // 第二次分发：根据访问者类型
    return builder->CreateAlloca(convertType(node.type)); 
}
```

2.**类型转换规则**

```c++
Type* CminusfBuilder::convertType(CminusType type) {
    switch(type) {
        case TYPE_INT: return INT32_T;
        case TYPE_FLOAT: return FLOAT_T;
        // ... 其他类型转换
    }
}
```

3.**作用域管理** 通过 `alloca`+ `load/store`实现变量覆盖等

```c++
void CminusfBuilder::visit(ASTFunDeclaration &node) {
    scope.enter(); // 进入函数作用域
    // 处理参数和函数体
    scope.exit();  // 退出作用域
}
```







**双重分发举例：**



**`param->accept(*this)` —— 第一次分发**



​	**调用者**: `visit(ASTFunDeclaration &)`。

​	**调用**: `param->accept(*this)`。

- `param`: 是一个指向 `ASTParam` 对象的指针。
- `*this`: 就是当前的 `CminusfBuilder` 对象，也就是“访问者”。

​	**发生了什么**:

- C++ 通过**虚函数机制 (virtual function)** 找到 `param` 指针所指向的**实际对象类型**——也就是 `ASTParam` 类。
- 然后，它调用 `ASTParam` 类中定义的 `accept` 方法。

```cpp
// In class ASTParam ast.cpp
Value* ASTParam::accept(ASTVisitor &visitor) {
    // 关键在这里！
    return visitor.visit(*this);
}
```

根据 `param` 的**运行时类型**，决定了执行哪个类的 `accept` 方法





`visitor.visit(*this)` —— 第二次分发



​	**调用者**: `ASTParam::accept` 方法。

​	**调用**: `visitor.visit(*this)`。

- `visitor`: 就是从第一步传进来的 `CminusfBuilder` 对象。
- `*this`: 在 `ASTParam` 类的成员函数里，`*this` 的**编译时类型**被精确地确定为 `ASTParam &`。

​	**发生了什么**:

- 编译器现在要选择 `visitor` (即 `CminusfBuilder` 对象) 上的 `visit` 方法来调用。
- `CminusfBuilder` 类中有很多个重载的 `visit` 方法

根据 `accept` 方法中 `this` 的**编译时类型**，决定了执行访问者（`CminusfBuilder`）的哪个 `visit` 重载版本



### 3.3 具体实现

`lightir`中`IRBuilder`封装了创建llvm IR的指令细节，是生成中间IR的核心工具

**——> 需要经常查看 `include/lightir/IRBuilder.hpp` 来了解它提供了哪些方法**

**`context`**是在 `CminusfBuilder` 里定义的一个结构体。因为访问者模式在遍历AST时，

我们使用的“访问者模式”框架，每个`visit`方法的签名都是固定的`Value* visit(ASTNode& node);`

返回值 `Value*` 并不总能满足我们的需求（比如，有时我们需要传递类型信息，有时需要传递计算结果）`context` 就像一个临时“快递箱”，用于在不同的`visit`方法之间传递信息。比如，`visit(ASTNum &node)` 会把创建的常量放到 `context.Num` 中，这样它的父节点 `visit(ASTAdditiveExpression &node)` 就可以从中取出这个常量来进行加法运算

**——> 需要用context作为工作台和记事本，解决AST节点间复杂的信息流转问题**



`CminusfBuilder` 类使用成员 `context` 存储翻译时状态，下列代码片段是 `context` 的定义，学生需要为该结构体添加更多域来存储翻译时的状态，接下来需要完成cminusf_builder.hpp

具体visit行为则需要完成cminusf_builder.cpp

**变量声明 (ASTVarDeclaration)**：支持全局变量、局部变量、数组和基本类型的声明。

**函数声明 (ASTFunDeclaration)**：正确处理函数签名，包括参数和返回类型。

**语句 (ASTStatement)**：

- **复合语句 (ASTCompoundStmt)**：正确处理作用域。
- **选择语句 (ASTSelectionStmt)**：生成`if-else`的跳转逻辑。
- **迭代语句 (ASTIterationStmt)**：生成`while`循环的跳转逻辑。
- **返回语句 (ASTReturnStmt)**：处理带返回值和`void`返回的情况，并进行必要的类型转换。

**表达式 (ASTExpression)**：

- **赋值 (ASTAssignExpression)**：处理赋值，包括类型转换。
- **二元运算 (ASTSimpleExpression, ASTAdditiveExpression, ASTTerm)**：支持整数和浮点数的混合运算，并进行适当的类型提升。
- **变量引用 (ASTVar)**：支持普通变量和数组元素的访问。
- **函数调用 (ASTCall)**：处理函数调用，包括参数的类型检查和转换。



* #### **cminusf_builder.hpp**

```c++
    struct {
        // function that is being built
        Function *func = nullptr;
        // TODO: you should add more fields to store state
        Value *Num = nullptr;
        Type *ParaType = nullptr;
        CminusType NumType = TYPE_VOID;
        Value *varAddr = nullptr;
        std::string param_id; 
        int count =0;
        int INTEGER=0;
    } context;
```



**字段解析**

1. **`std::string param_id`**:
   - 用于存储当前参数的标识符（名称）
   - 在处理函数参数时，记录参数的名称以便后续使用
2. **`int count`**:
   - 用于计数
3. **`int INTEGER`**:
   - 用于存储当前整数值
   - 在数组声明时，记录数组的大小
4. 

|    成员    |     类型     |                作用                |
| :--------: | :----------: | :--------------------------------: |
| `NumType`  | `CminusType` |   保存当前操作/表达式的类型信息    |
|   `Num`    |   `Value*`   | 保存当前操作/表达式的值（IR 表示） |
| `varAddr`  |   `Value*`   |         保存变量访问的地址         |
| `ParaType` |   `Type*`    |            保存参数类型            |






* #### **cminusf_builder.cpp**

  `cminusf_builder.cpp` 中的行为需要根据 `ast.hpp` 中定义的 AST 结构体和**cminusf提供的语法规则**进行修改和实现。

  

  * **ASTNum**

    `ASTNum` 结构体的 `accept` 方法会调用 `ASTVisitor`，而 `ASTVisitor` 的实现需要根据 `ASTNum` 的 `type` 和具体的值（`i_val` 或 `f_val`）生成对应的 IR

    ```cpp
    // ast.hpp
    struct ASTNode {
        virtual Value *accept(ASTVisitor &) = 0;
        virtual ~ASTNode() = default;
    };
    
    struct ASTFactor : ASTNode {
        virtual ~ASTFactor() = default;
    };
    
    struct ASTNum : ASTFactor {
        virtual Value *accept(ASTVisitor &) override final;
        CminusType type;
        union {
            int i_val;
            float f_val;
        };
    };
    ```

    定义了type

    
    
    ​	**思考**: 当我们遇到一个`ASTNum`节点时，我们要做什么？
    
    ​	**回答**: 我们需要创建一个LLVM IR的常量。
    
    ​	**行动**:
    
    ​		判断是整数还是浮点数。
    
    ​		调用`ConstantInt::get()`或`ConstantFP::get()`来创建常量。
    
    ​		把创建好的常量存到`context`里，供其他`visit`方法使用。
    
    
    
    先定义宏
    
    ```cpp
    #define CONST_FP(num) ConstantFP::get((float)num, module.get())
    #define CONST_INT(num) ConstantInt::get(num, module.get())
    ```
    
    
    
    ```cpp
    Value* CminusfBuilder::visit(ASTNum &node) {
        // TODO: This function is empty now.
        // Add some code here.
        if (node.type == TYPE_INT) {
            context.NumType = TYPE_INT;
            context.Num = CONST_INT(node.i_val);
            context.INTEGER = node.i_val;
        }
        else if (node.type == TYPE_FLOAT) {
            context.NumType = TYPE_FLOAT;
            context.Num = CONST_FP(node.f_val);
        }
        else {
            context.NumType = TYPE_VOID;
            context.Num = nullptr;
        }
        return nullptr;
    }
    ```
    
    其中`INTEGER`存储，对于数组获取int值情况更便捷
  
  

  

  * **ASTVarDeclaration**

    ```cpp
    // ast.hpp
    struct ASTDeclaration : ASTNode {
        virtual ~ASTDeclaration() = default;
        CminusType type;
        std::string id;
    };
    
    struct ASTVarDeclaration : ASTDeclaration {
        virtual Value *accept(ASTVisitor &) override final;
        std::shared_ptr<ASTNum> num;
    };
    ```
  
    `std::shared_ptr<ASTNum> num;`	ASTNum类型指针，表示变量的维度信息
  
    ` CminusType type;`
  
    ` std::string id;`
    接下来根据语法规则：
  
    > var-declaration →type-specifier **ID** **;** ∣ type-specifier **ID** **[** **INTEGER** **]** ;
    >
    > type-specifier→**int** ∣ **float** ∣ **void**
  
    
  
    所以需要判断变量和类型
  
    除此之外，我们需要考虑定义的变量是全局变量吗，它的生命周期
  
    - 全局变量需要在 LLVM IR 中声明为 `GlobalVariable`，它们在程序的整个生命周期内都存在。局部变量需要使用 `alloca` 指令分配在栈上，生命周期仅限于当前函数。
    - 而LLVM IR 是编译器的中间表示，必须准确反映变量的作用域和存储方式；因此需要管理scope作用域，方便变量的查找和使用
  
    - 使用 `scope.in_global()` 方法判断当前变量是否在全局作用域中；
    - 全局变量调用 `GlobalVariable::create`，(全局内存)并初始化为零值，局部变量调用 `builder->create_alloca`，在栈上分配内存
    - 调用 `scope.push(node.id, varAlloca)` 将变量存储到作用域中
  
    
  
    **全局变量跳转GlobalVariable定义查看参数设置:**
  
    `  GlobalVariable(std::string name, Module m, Type ty, bool is_const, Constant init = nullptr);`
  
    ```cpp
        static GlobalVariable *create(std::string name, Module *m, Type *ty,
                                      bool is_const, Constant *init);
    ```
  
    `node.id` 的值是在语法分析阶段解析源代码时确定的，所以不需要也不能显示修改。
  
    `static ConstantZero *get(Type ty, Module m)` 表示0初始化 ——>	`ConstantZero::get(type, module.get())`
  
    **局部变量跳转create_alloca定义查看参数设置:**
  
    ```cpp
        AllocaInst *create_alloca(Type *ty) {
            return AllocaInst::create_alloca(ty, this->BB_);
        }
    ```
  
    只需要type即可
  
    
  
    ​	**思考**: 声明一个变量意味着什么？
  
    ​	**回答**: 意味着需要在内存里为它分配一块空间。
  
    ​	**行动**:
  
    ​		确定变量的类型（`int`, `float`, 还是数组）。
  
    ​		**全局变量 vs. 局部变量**:
  
    ​			如果`scope.in_global()`为真，说明在所有函数之外，是全局变量。使用 `GlobalVariable::create()` 创建。
  
    ​			否则，是函数内的局部变量。使用 `builder->create_alloca()` 在函数的栈帧上分配空间。
  
    ​		把变量名和它分配到的内存地址记录到当前作用域中：`scope.push(node.id, alloc)`。
  
    
  
    **流程如下：**
  
    * 首先根据`node.type`判断变量的类型：
  
      ​	如果是 `TYPE_INT`，将类型设置为 `INT32_T`；如果是 `TYPE_FLOAT`，将类型设置为 `FLOAT_T`
  
    * 使用 `scope.in_global()` 判断当前是否在全局作用域:
  
      ```cpp
      if (not scope.in_global())
      {
          // 局部作用域处理
      }
      else
      {
          // 全局作用域处理
      }
      ```
  
    * 局部:
  
      **判断是否是数组声明**：
  
      - 如果`node.num == nullptr`，说明不是数组：
  
        创建一个普通变量的局部分配（`create_alloca`）,将变量加入作用域（`scope.push`）
  
      - 如果`node.num != nullptr`，说明是数组：
  
        ​	调用 `node.num->accept(*this)` 计算数组大小。
  
        ​	如果数组大小 `context.INTEGER <= 0`，调用异常处理函数 `neg_idx_except`。
  
        ​	创建数组类型（`ArrayType::get`）并分配内存（`create_alloca`），最后加入作用域
  
        ​		【`ArrayType::get(tp, context.INTEGER)` 是一个静态方法，用于生成一个数组类型对象】
  
        **此处用`context.INTEGER`而不是`context.Num`：因为 context.Num是IR 表示，value* 类型，用于数组索引**
  
    * 全局同理
  
    
  
    
  
    ```cpp
    Value* CminusfBuilder::visit(ASTVarDeclaration &node) {
        // TODO: This function is empty now.
        // Add some code here.
        Type *tp = nullptr;
        if (node.type == TYPE_INT)
        {
            tp = INT32_T;
        }
        else if (node.type == TYPE_FLOAT)
        {
            tp = FLOAT_T;
        }
        else {
            return nullptr;
        }
    
        if (not scope.in_global())
        {
            if (node.num == nullptr)
            {
                auto Alloca = (tp != nullptr) ? builder->create_alloca(tp) : nullptr;
                scope.push(node.id, Alloca);
            }
            else
            {
                node.num->accept(*this);
                if (context.INTEGER <= 0)
                    builder->create_call(scope.find("neg_idx_except"), std::vector<Value *>{});
                auto arrytype = ArrayType::get(tp, context.INTEGER);
                auto arryAllca = builder->create_alloca(arrytype);
                scope.push(node.id, arryAllca);
            }
        }
        else
        {
            auto initializer = ConstantZero::get(tp, builder->get_module());
            if (node.num == nullptr)
            {
    
                auto Alloca = (tp != nullptr) ? GlobalVariable::create(node.id, builder->get_module(), tp, false, initializer) : nullptr;
                scope.push(node.id, Alloca);
            }
            else
            {
                node.num->accept(*this);
                if (context.INTEGER <= 0)
                    builder->create_call(scope.find("neg_idx_except"), std::vector<Value *>{});
                auto arrytype = ArrayType::get(tp, context.INTEGER);
                auto arryAllca = GlobalVariable::create(node.id, builder->get_module(), arrytype, false, initializer);
                scope.push(node.id, arryAllca);
            }
        }
        return nullptr;
    }
    ```
  
  
  
  
  
  
  
  
  * **ASTParam**
  
    ```cpp
    struct ASTParam : ASTNode {
        virtual Value *accept(ASTVisitor &) override final;
        CminusType type;
        std::string id;
        // true if it is array param
        bool isarray;
    };
    ```
  
    `isarray`判断数组
  
    再看语法:
  
    > param→type-specifier **ID** ∣ type-specifier **ID** **[** **]**
  
    
  
    **思考**:
  
    - 这个函数的**核心任务**是什么？
    - 根据我们之前在 `visit(ASTFunDeclaration &)` 中的分析，这个函数被调用时，它的上级（`visit(ASTFunDeclaration &)`）需要知道两件事：**1. 这个参数在 LLVM IR 中的具体类型是什么？** **2. 这个参数的名字是什么？** 这个函数必须把这两个信息准备好，并传递回去
  
    **回答**:
  
    - 要完成这个任务，我需要检查 `ASTParam` 节点 (`node`) 的两个关键属性：`node.type` (是 `TYPE_INT` 还是 `TYPE_FLOAT`) 和 `node.isarray` (布尔值，表示是否是数组)
    - 根据这两个属性的组合，我可以确定出四种可能的 IR 类型
    - 因为不能通过 `return` 返回信息，我必须将计算出的 **IR 类型**和**参数名**存入共享的 `context` 对象中，以便上级函数可以从中读取
  
    **行动**:
  
    1. **传递名字**: 不论类型是什么，参数的名字都是 `node.id`。所以，第一步就是 `context.param_id = node.id;`，将名字存入 `context`
    2. **判断是否为数组**: `node.isarray` 是最主要的分类依据
       - **如果是数组** (例如 `int a[]`)：在 C 语言中，数组作为函数参数时，会**退化成指向其首元素的指针**。因此，我们需要创建一个指针类型
         - 如果 `node.type` 是 `TYPE_INT`，那么 IR 类型就是“指向 i32 的指针”，通过 `PointerType::get(INT32_T)` 创建
         - 如果 `node.type` 是 `TYPE_FLOAT`，那么 IR 类型就是“指向 float 的指针”，通过 `PointerType::get(FLOAT_T)` 创建
       - **如果不是数组** (例如 `int a`)：直接使用对应的基础类型
         - 如果 `node.type` 是 `TYPE_INT`，那么 IR 类型就是 `INT32_T`
         - 如果 `node.type` 是 `TYPE_FLOAT`，那么 IR 类型就是 `FLOAT_T`
    3. **存储类型**: 将最终确定好的 `Type*` 对象存入 `context.ParaType` 中，完成向上级函数的信息传递 
  
    
  
    ```cpp
    Value* CminusfBuilder::visit(ASTParam &node) {
        // TODO: This function is empty now.
        // Add some code here.
         context.param_id=node.id;
        if (node.isarray)
        {
            if (node.type == TYPE_INT)
            {
                context.ParaType = PointerType::get(INT32_T);
            }
            else if (node.type == TYPE_FLOAT)
            {
                context.ParaType = PointerType::get(FLOAT_T);
            }
            else
            {
                context.ParaType = PointerType::get(VOID_T);
            }
        }
        else
        {
            if (node.type == TYPE_INT)
            {
                context.ParaType = INT32_T;
            }
            else if (node.type == TYPE_FLOAT)
            {
                context.ParaType = FLOAT_T;
            }
            else
            {
                context.ParaType = VOID_T;
            }
        }
        return nullptr;
    }
    ```
  
  `PointerType::get()`是 LLVM 中创建指针类型的方法
  
  如：INT ——> INT *
  
  
  
  * **ASTFunDeclaration**
  
    > fun-declaration→type-specifier **ID** **(** params **)** compound-stmt
    >
    > type-specifier→**int** ∣ **float** ∣ **void**
    >
    > params→param-list ∣ **void**
    >
    > param-list→param-list **,** param ∣ param
    >
    > param→type-specifier **ID** ∣ type-specifier **ID** **[** **]**
    >
    > ...
  
    添加向量param_id
  
    ```cpp
    std::vector<Type *> param_types;
    std::vector<std::string> param_id;
    ```
  
    这里注意Type * 传递，所以需要context中的`Type *ParaType = nullptr;`存储翻译
  
    
  
    **第一个todo，需要从每个参数节点中提取类型信息，添加到`param_types`向量中，用于构造函数签名:**
    
    **目标是什么？** 我的目标是遍历函数的所有参数 (`node.params`)，为每一个参数确定它在 **LLVM IR 中对应的具体类型**，然后把这个类型 (`Type*`) 收集到 `param_types` 这个 vector 中。同时，我也需要记下每个参数的名字，存入 `param_id`，供后面使用。
    
    **如何确定每个参数的 IR 类型？**
    
    1. **分派任务**：参数的类型信息（是 `int` 还是 `float`？是不是数组？）都封装在 `ASTParam` 节点里。直接在当前函数里写一堆 `if-else` 来判断是不优雅的。最好的方法是遵循**访问者模式**，调用 `param->accept(*this)`，将这个具体的任务分派给 `visit(ASTParam &)` 函数去处理。
    2. **取回信息**：在 `param->accept(*this)` 调用返回后，我们就可以确信 `context.ParaType` 中已经存放了正确的参数类型。我们只需 `param_types.push_back(context.ParaType)` 就能把它收集起来。同理，我们让 `visit(ASTParam&)` 也把参数名存入 `context.param_id`，然后在这里取回。
    
    
    
    **第二个todo，处理参数并存入作用域**
    
    **目标是什么？** 函数定义中的参数（如 `%a`, `%b`）是传入的**值**，它们是只读的，并且没有内存地址。但在 C/Cminusf 中，我们可以像修改局部变量一样修改参数（例如 `a = a + 1;`）。为了实现这一点，我们必须为每个参数在**函数栈帧上分配一块内存**，将传入的值**复制**进去，然后让函数体内的代码通过参数名访问这块内存。
    
    **具体操作步骤是什么？** 这是一个经典的三部曲，对应三条 IR 指令：
    
    1. **分配内存 (`alloca`)**：为第 `i` 个参数分配一块内存空间。分配多大的空间呢？当然是和这个参数的类型 (`args[i]->get_type()`) 一样大。这通过 `builder->create_alloca()` 实现。
       - **IR 效果**: `%a_addr = alloca i32`
    2. **存储值 (`store`)**：内存分配好了，现在是空的。我们需要把传入的参数值 `args[i]` 存到刚刚分配好的内存地址 `argAlloca` 中。这通过 `builder->create_store()` 实现。
       - **IR 效果**: `store i32 %a, i32* %a_addr`
    3. **注册到符号表 (`scope.push`)**：现在，参数 `a` 在函数体内的“实体”就是这个内存地址 `%a_addr`。我们必须将参数名 `param_id[i]` 和这个地址 `argAlloca` 关联起来，并存入当前作用域。这样，当函数体中出现 `a` 时，我们才能通过 `scope.find("a")` 找到它的内存地址。
    
    
    
    ```cpp
    for (auto &param : node.params)
        {
            // TODO: Please accomplish param_types.
            param->accept(*this);
            param_types.push_back(context.ParaType);
            param_id.push_back(context.param_id);
        }
        
        
    for (int i = 0; i < node.params.size(); ++i)
        {
            // TODO: You need to deal with params and store them in the scope.
            auto argAlloca = builder->create_alloca(args[i]->get_type());
            builder->create_store(args[i], argAlloca);
            scope.push(param_id[i], argAlloca);
        }
    ```
  
  
  
  
  
  * **ASTCompoundStm**
  
    > compound-stmt→**{** local-declarations statement-list **}**
    >
    > local-declarations→local-declarations var-declaration ∣ empty
    >
    > statement-list→statement-list statement ∣ empty
    >
    > statement→ expression-stmt∣ compound-stmt∣ selection-stmt∣ iteration-stmt∣ return-stmt
  
  ```cpp
  struct ASTStatement : ASTNode {
      virtual ~ASTStatement() = default;
  };
  
  struct ASTCompoundStmt : ASTStatement {
      virtual Value *accept(ASTVisitor &) override final;
      std::vector<std::shared_ptr<ASTVarDeclaration>> local_declarations;
      std::vector<std::shared_ptr<ASTStatement>> statement_list;
  };
  ```
  
  ```cpp
  Value* CminusfBuilder::visit(ASTCompoundStmt &node) {
      // TODO: This function is not complete.
      // You may need to add some code here
      // to deal with complex statements.
      scope.enter();
  
      for (auto &decl : node.local_declarations) {
          decl->accept(*this);
      }
  
      for (auto &stmt : node.statement_list) {
          stmt->accept(*this);
          if (builder->get_insert_block()->is_terminated())
              break;
      }
  
      scope.exit();
      return nullptr;
  }
  ```
  
  ​	加入`scope.enter() scope.exit()`
  
  ​	**思考**:
  
  ​		当我们遇到一个 `{...}` 代码块时，它在编程语言中意味着什么？
  
  ​		它意味着一个**新的、独立的作用域**。在这个代码块里声明的变量（局部变量）只在这个块内部有效。一旦程序执行离开这个		代码块，这些变量就应该被销毁，它们的名字在外部也就不再可见。
  
  ​		因此，`visit(ASTCompoundStmt &)` 的核心职责就是**管理作用域的生命周期**，并按顺序处理其中的内容
  
  ​	**回答**:
  
  ​		要正确地处理一个复合语句，我必须严格遵循以下流程：
  
  ​			**进入新作用域**：在处理这个代码块的任何内容之前，我必须先通知我的符号表管理器（`scope` 对象），我们进入了一			个新的层级。	
  
  ```c++
  scope.enter();
  ```
  
  ​			**处理局部变量声明**：我必须先遍历并处理所有的 `local-declarations`。这样，这些局部变量才会被注册到刚刚创建			的新作用域中，供后面的语句使用。
  
  ```		cpp
      for (auto &decl : node.local_declarations) {
          decl->accept(*this);
      }
  ```
  
  ​			**处理语句列表**：在所有局部变量都声明完毕后，我再按顺序遍历并处理 `statement-list` 中的每一条语句。
  
  ​			// **如果遇到 return 等终止指令，后续的语句都是死代码，必须停止生成**
  
  ```c++
      for (auto &stmt : node.statement_list) {
          stmt->accept(*this);
          if (builder->get_insert_block()->is_terminated())
              break;
      }
  ```
  
  ​			**退出作用域**：当这个代码块中所有的语句都处理完毕后，我必须通知符号表管理器，我们要退出当前作用域。这会自			动“销毁”所有在这个作用域中声明的变量，防止它们“泄露”到外部。
  
  ```c++
  scope.exit();
  ```
  
  
  
  
  
  - **ASTExpressionStmt**
  
    > expression-stmt→expression ; ∣ ;
  
    ```c++
    struct ASTExpressionStmt : ASTStatement {
        virtual Value *accept(ASTVisitor &) override final;
        std::shared_ptr<ASTExpression> expression;
    };
    ```
  
    检查是否存在表达式，如果不存在则返回nullptr不处理即不生成相应IR
  
    ```cpp
    Value* CminusfBuilder::visit(ASTExpressionStmt &node) {
        // TODO: This function is empty now.
        // Add some code here.
        
        if(node.expression != nullptr){
            node.expression -> accept(*this);
        }
        
        return nullptr;
    }
    ```
  
    
  
  
  
  
  
  - **ASTSelectionStmt**
  
    **这个函数负责处理 `if-else` 语句**
  
    > selection-stmt→ **if** **(** expression **)** statement
    >
    > ​			     ∣ **if** **(** expression **)** statement **else** statement
  
    `if` 语句中的表达式将被求值，若结果的值等于 0，则第二个语句执行（如果存在的话），否则第一个语句会执行
  
    ```c++
    struct ASTSelectionStmt : ASTStatement {
        virtual Value *accept(ASTVisitor &) override final;
        std::shared_ptr<ASTExpression> expression;
        std::shared_ptr<ASTStatement> if_statement;
        // should be nullptr if no else structure exists
        std::shared_ptr<ASTStatement> else_statement;
    };
    ```
  
    **思考**:
  
    ​	**LLVM IR 中如何表达“选择”？** IR 中没有高级的 `if` 关键字。它使用更底层的部件来模拟这个行为：
  
    ​		**基本块 (Basic Block)**：每一段可能被执行的代码（`if` 里的代码、`else` 里的代码、`if-else` 之后汇合的代码）都需		要放在一个独立的基本块中。
  
    ​		**条件跳转指令 (`br`)**: 在计算完条件后，需要一条指令，根据条件的真 (`true`) 或假 (`false`)，**跳转**到不同的基本		块。
  
    ​		**无条件跳转指令 (`br`)**: 在 `if` 分支执行完后，需要一条指令**无条件地跳过** `else` 分支，直接到达汇合点。
  
    ![a flowchart for an if-else statement的图片](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcQSC_OrbsN9OxuSdy-4P4FX0QfpYFrwZoy1OPh1UA4xhek5EBbfQe-4IctIpAAjHq8UJtzaTNBd42WMjKDNu6u2bQUBOEyHdZNZnT84qH06GwhmNU8)
  
    **回答**:
  
    ​	**生成计算条件的代码**：首先，处理 `if` 括号里的表达式。
  
    ```c++
        if(node.expression != nullptr){
            node.expression -> accept(*this);
        }
    ```
  
    ​	**将条件结果转为布尔值**：LLVM 的条件跳转指令需要一个 `i1` 类型（1位整数，即 `true/false`）的布尔值。而 Cminusf 的	表达式结果通常是 `i32` 或 `float`。因此，需要一个额外的比较指令（例如 `!= 0`）来完成这个转换。
  
    ```cpp
        Value *cond_val = context.Num;
        if(context.NumType == TYPE_INT){
            cond_val = builder-> create_icmp_ne(cond_val, CONST_INT(0));
        }
        else if(context.NumType == TYPE_FLOAT){
            cond_val = builder-> create_fcmp_ne(cond_val, CONST_FP(0.0));
        }
        else{
            cond_val = nullptr;
        }
    ```
  
    ​	**创建基本块**：我需要为 `if` 分支（`true_bb`）、`else` 分支（`false_bb`）以及 `if-else` 结束后的汇合点（`exit_bb`）分别	创建基本块。
  
    ```c++
        auto *func = context.func;
        auto *true_bb = BasicBlock::create(module.get(), "if.true" + std::to_string(context.count++), func);
        auto *false_bb = BasicBlock::create(module.get(), "if.false" + std::to_string(context.count++), func);
        auto *exit_bb = BasicBlock::create(module.get(), "if.exit" + std::to_string(context.count++), func);
    ```
  
    ​	**连接基本块**：使用跳转指令将这些基本块按照正确的逻辑顺序“连接”起来。
  
    ```c++
        if(node.else_statement){
            builder-> create_cond_br(cond_val, true_bb, false_bb);
        }
        else {
            builder-> create_cond_br(cond_val, true_bb, exit_bb);
        }
    ```
  
    ​	**填充基本块**：递归地调用 `accept` 方法，为 `if` 和 `else` 的语句体生成代码，并填充到对应的基本块中。
  
    ```c++
        builder-> set_insert_point(true_bb);
        node.if_statement -> accept(*this);
        if(!builder->get_insert_block()->is_terminated()){
            builder-> create_br(exit_bb);
        }
    
        if(node.else_statement){
            builder-> set_insert_point(false_bb);
            node.else_statement -> accept(*this);
            if(!builder->get_insert_block()->is_terminated()){
                builder-> create_br(exit_bb);
            }
        }
        else {
            false_bb->erase_from_parent();
        }
        
        builder-> set_insert_point(exit_bb);
    ```
  
    
  
  
  
  ```cpp
  Value* CminusfBuilder::visit(ASTSelectionStmt &node) {
      // TODO: This function is empty now.
      // Add some code here.
  
      if(node.expression != nullptr){
          node.expression -> accept(*this);
      }
      Value *cond_val = context.Num;
      if(context.NumType == TYPE_INT){
          cond_val = builder-> create_icmp_ne(cond_val, CONST_INT(0));
      }
      else if(context.NumType == TYPE_FLOAT){
          cond_val = builder-> create_fcmp_ne(cond_val, CONST_FP(0.0));
      }
      else{
          cond_val = nullptr;
      }
  
      auto *func = context.func;
      auto *true_bb = BasicBlock::create(module.get(), "if.true" + std::to_string(context.count++), func);
      auto *false_bb = BasicBlock::create(module.get(), "if.false" + std::to_string(context.count++), func);
      auto *exit_bb = BasicBlock::create(module.get(), "if.exit" + std::to_string(context.count++), func);
  
      if(node.else_statement){
          builder-> create_cond_br(cond_val, true_bb, false_bb);
      }
      else {
          builder-> create_cond_br(cond_val, true_bb, exit_bb);
      }
  
      builder-> set_insert_point(true_bb);
      node.if_statement -> accept(*this);
      if(!builder->get_insert_block()->is_terminated()){
          builder-> create_br(exit_bb);
      }
  
      if(node.else_statement){
          builder-> set_insert_point(false_bb);
          node.else_statement -> accept(*this);
          if(!builder->get_insert_block()->is_terminated()){
              builder-> create_br(exit_bb);
          }
      }
      else {
          false_bb->erase_from_parent();
      }
      
      builder-> set_insert_point(exit_bb);
  
      return nullptr;
  }
  ```
  
  `context.count++` 在这里的核心作用是**为基本块 (Basic Block) 生成独一无二的名字**
  
  
  
  - 

