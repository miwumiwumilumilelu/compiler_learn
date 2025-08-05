> # _lab2 Light IR C++中间代码生成
>

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



### 3.3 具体实现

`CminusfBuilder` 类使用成员 `context` 存储翻译时状态，下列代码片段是 `context` 的定义，学生需要为该结构体添加更多域来存储翻译时的状态，接下来需要完成cminusf_builder.hpp

具体visit行为则需要完成cminusf_builder.cpp
