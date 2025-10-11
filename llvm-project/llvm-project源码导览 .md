# llvm-project项目源码总览

![tree](./img/tree.png)

> .
> ├── llvm/         # 核心库：IR、优化器、后端
> ├── clang/        # C/C++/Objective-C 的前端
> ├── lld/          # LLVM 的链接器，速度极快
> ├── lldb/         # LLVM 的调试器
> ├── compiler-rt/  # 编译器运行时库（底层函数支持）
> ├── libcxx/       # C++ 标准库 (std::vector, std::string 等)
> ├── clang-tools-extra/ # 基于 Clang 的额外工具 (静态分析、重构)
> ├── mlir/         # (新) 多级中间表示，下一代编译器基础设施
> └── ... (其他子项目如 openmp, polly 等)

* **llvm/ - 项目的心脏**
  这是 LLVM 的核心。如果你想理解 LLVM 的工作原理，这里是起点。
  * llvm/include/llvm/IR/: 定义了 LLVM IR 的核心数据结构，比如 Instruction, BasicBlock, Function, Module。这是理解一切的基础。
  * llvm/lib/IR/: 上述头文件的实现。
  * llvm/lib/Transforms/: 优化器的所在地。它分为 Scalar（标量优化，针对单个指令）、IPO（过程间优化）、Vectorize（向量化）等。每个优化都是一个独立的 "Pass"（遍）。你可以像插拔插件一样组合这些 Pass。
  * llvm/lib/Target/: 后端的所在地。每个子目录对应一个目标平台，比如 X86, AArch64 (ARM64), RISCV。这里实现了如何将 LLVM IR 转换为特定平台的汇编代码。
  * llvm/tools/: 存放一些命令行工具，比如 llc (LLVM 静态编译器) 和 lli (LLVM IR 解释器)。

* **clang/ - 最著名的前端**
  Clang 是 C、C++、Objective-C 和 Objective-C++ 的前端。
  * clang/lib/Lex/: 词法分析器 (Lexer)，将源码变成一个个 Token。
  * clang/lib/Parse/: 语法分析器 (Parser)，将 Token 组合成一棵抽象语法树 (Abstract Syntax Tree, AST)。
  * clang/lib/Sema/: 语义分析器 (Semantic Analyzer)，检查 AST 的语法正确性（比如类型检查）。
  * clang/lib/CodeGen/: 代码生成器，负责将 AST 转换成 LLVM IR。这是 Clang 和 LLVM 核心库的桥梁。
  * clang/tools/driver/: 驱动程序，也就是你命令行里敲的 clang 命令本身。它负责解析命令行参数，并按顺序调用词法分析、解析、代码生成、链接等步骤。

* **lld/ - 高性能链接器**
  一个现代化的、极速的链接器，旨在替代系统默认的 ld。它的速度非常快，对于大型 C++ 项目能显著缩短链接时间。它有不同的版本，比如 ld.lld (for ELF/Linux), lld-link (for COFF/Windows), ld64.lld (for Mach-O/macOS)。
* **lldb/ - 现代化的调试器**
  类似于 GDB，但深度集成了 Clang 和 LLVM 的库。这让它能更好地理解 C++ 的复杂类型、模板等。Xcode 的默认调试器就是 LLDB。
  * compiler-rt/, libcxx/, libunwind, libcxxabi
    这些是支持 C++ 程序运行所必需的“全家桶”。
  * compiler-rt: 提供编译器需要的底层运行时函数，比如整数溢出检查、浮点数转换、原子操作等。
  * libcxx: 一个高性能的 C++ 标准库实现。
  * libcxxabi / libunwind: 提供 C++ ABI (Application Binary Interface) 支持，如异常处理、RTTI (运行时类型信息) 和栈展开 (unwinding)。

* **clang-tools-extra/**
  这完美体现了 LLVM 的“工具箱”哲学。由于 Clang 是一个库，任何人都可以用它的 AST 来构建强大的工具。
  * clang-tidy: 一个可扩展的静态分析框架，用来检查代码风格、潜在 Bug、性能问题等。
  * clangd: 一个基于 LSP (Language Server Protocol) 的服务，为 VS Code, Vim 等编辑器提供代码补全、定义跳转、重构等 IDE 功能。
  * clang-format: 代码格式化工具。

* **mlir/ - 面向未来的架构**
  MLIR (Multi-Level Intermediate Representation) 是一个更通用、更灵活的编译器基础设施，可以看作是 LLVM IR 的演进版。它特别适合处理领域特定语言（DSL）和异构计算（如 CPU+GPU+TPU）。它是当前编译器领域最热门的研究方向之一。

  