# _lab3 后端代码生成

**经过 Lab1 和 Lab2，编译器能够将 Cminusf 源代码翻译成 Light IR**

**本次实验要求将 IR 翻译成龙芯汇编指令**

> .
> ├── ...
> ├── include
> │   ├── ...
> │   └── codegen
> │       ├── ASMInstruction.hpp  # 描述汇编指令
> │       ├── CodeGen.hpp         # 后端框架顶层设计
> │       ├── CodeGenUtil.hpp     # 一些辅助函数及宏的定义
> │       └── Register.hpp        # 描述寄存器
> ├── src
> │   ├── ...
> │   └── codegen
> │       ├── CMakeLists.txt
> │       ├── CodeGen.cpp     <-- lab3 第二阶段需要修改的文件
> │       └── Register.cpp
> └── tests
>     ├── ...
>     └── 3-codegen
>         ├── warmup          <-- lab3 第一阶段（代码撰写）
>         └── autogen         <-- lab3 第二阶段的测试



## 1.阶段1：warmup预热实验

> .
> ├── ...
> ├── include
> │   ├── common
> │   └── codegen/*
> └── tests
>     ├── ...
>     └── 3-codegen
>         └── warmup
>             ├── CMakeLists.txt
>             ├── ll_cases          <- 需要翻译的 ll 代码
>             └── stu_cpp           <- 学生需要编写的汇编代码手动生成器

**实验内容**

实验在 `tests/3-codegen/warmup/ll_cases/` 目录下提供了六个 `.ll` 文件。

需要在 `tests/3-codegen/warmup/stu_cpp/` 目录中，依次完成 `assign_codegen.cpp`、`float_codegen.cpp`、`global_codegen.cpp`、`function_codegen.cpp`、`icmp_codegen.cpp` 和 `fcmp_codegen.cpp` 六个 C++ 程序中的 TODO。

这六个程序运行后应该能够生成 `tests/3-codegen/warmup/ll_cases/` 目录下六个 `.ll` 文件对应的汇编程序。





## 2.阶段2：编译器后端

一个典型的编译器后端从中间代码获取信息，进行**活跃变量分析、寄存器分配、指令选择、指令优化**等一系列流程，最终生成高质量的后端代码。

本次实验，这些复杂的流程被简化，仅追求实现的完整性，要求采用**栈式分配的策略**，完成后端代码生成。

> .
> ├── include
> │   └── codegen/*                   # 相关头文件
> ├── src
> │   └── codegen
> │       └── CodeGen.cpp         <-- 学生需要补全的文件
> └── tests
>     ├── 3-codegen
>     │   └── autogen
>     │       ├── eval_lab3.sh    <-- 测评脚本
>     │       └── testcases       <-- lab3 第二阶段的测例目录一
>     └── testcases_general       <-- lab3 第二阶段的测例目录二

**实验内容**

补全 `src/codegen/CodeGen.cpp` 中的 TODO，并按需修改 `include/codegen/CodeGen.hpp` 等文件，使编译器能够生成正确的汇编代码。