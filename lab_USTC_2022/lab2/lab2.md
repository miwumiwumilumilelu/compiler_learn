
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



## 2. 访问者模式
