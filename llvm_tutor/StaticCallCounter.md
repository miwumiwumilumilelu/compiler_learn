# StaticCallCounter

## StaticCallCounter Pass 使用 —— 静态函数分析

分析类Pass，**在编译时**统计代码中有多少个直接的函数调用

静态是指这些函数调用是编译时调用（即在编译期间可见）。这与动态函数调用（即在运行时（编译后的模块运行时）遇到的函数调用）形成对比。在分析循环内的函数调用时，这种区别变得显而易见，例如：

```c++
  for (i = 0; i < 10; i++)
    foo();
```

虽然在运行时 `foo` 会被执行 10 次， 但 StaticCallCounter 将仅报告 1 个函数调用

此过程仅考虑直接函数调用，不考虑通过函数指针进行的函数调用



1. 方法一：通过 `opt` 工具运行

   **生成 Bitcode 文件** `input_for_cc.bc` 文件

   ```shell
   ~/projects/llvm-project/build/bin/clang -emit-llvm -c ../inputs/input_for_cc.c -o input_for_cc.bc
   ```

   使用 `opt` 工具加载 `StaticCallCounter` 插件并执行它

   ```shell
   llvm-tutor/build on  main [?] via 🅒 base 
   ➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libStaticCallCounter.dylib -passes="print<static-cc>" -disable-output input_for_cc.bc
   =================================================
   LLVM-TUTOR: static analysis results
   =================================================
   NAME                 #N DIRECT CALLS
   -------------------------------------------------
   foo                  3         
   bar                  2         
   fez                  1         
   -------------------------------------------------
   ```

2. 方法二：通过独立的 `static` 工具运行

   `llvm-tutor` 项目提供了一个名为 `static` 的独立命令行工具，它是 `StaticCallCounter` Pass 的一个**专属包装器 (Wrapper)**。使用它会更简单，因为它内部已经处理好了 Pass 的加载和运行

   ```shell
   llvm-tutor/build on  main [?] via 🅒 base 
   ➜ ./bin/static input_for_cc.bc
   =================================================
   LLVM-TUTOR: static analysis results
   =================================================
   NAME                 #N DIRECT CALLS
   -------------------------------------------------
   foo                  3         
   bar                  2         
   fez                  1         
   -------------------------------------------------
   ```



## StaticCallCounter 源码

### .h

用`MapVector`来存储：**被调用函数的指针 (`const llvm::Function \*`) -> 该函数被调用的次数 (`unsigned`)**

`struct StaticCallCounter : public llvm::AnalysisInfoMixin<StaticCallCounter>{}`结构体需要考虑run函数处理的单元应该是Module；以及分析类对应的key`static llvm::AnalysisKey Key;`

`class StaticCallCounterPrinter: public llvm::PassInfoMixin<StaticCallCounterPrinter> {}`类应该考虑输出流&OS

### .c

```c++
StaticCallCounter::Result StaticCallCounter::runOnModule(Module &M) {
  llvm::MapVector<const llvm::Function *, unsigned> Res;

  for (auto &Func : M) {      // 遍历模块中的每个函数
    for (auto &BB : Func) {   // 遍历函数中的每个基本块
      for (auto &Ins : BB) {  // 遍历基本块中的每条指令

        // 尝试将指令转换为一个调用指令 (CallBase)
        auto *CB = dyn_cast<CallBase>(&Ins);
        if (nullptr == CB) {
          continue; // 如果不是调用指令，则跳过
        }

        // 获取被直接调用的函数
        auto DirectInvoc = CB->getCalledFunction();
        if (nullptr == DirectInvoc) {
          continue; // 如果是间接调用 (如通过函数指针)，则跳过
        }

        // ... 更新计数器 ...
        auto CallCount = Res.find(DirectInvoc);
        if (Res.end() == CallCount) {
          CallCount = Res.insert(std::make_pair(DirectInvoc, 0)).first;
        }
        ++CallCount->second;
      }
    }
  }

  return Res;
}
```

`dyn_cast` 会尝试进行安全的类型转换，如果当前指令 `Ins` 不是一个调用指令，它会返回 `nullptr`

**区分直接/间接调用**：`CB->getCalledFunction()` 用于获取被调用的 `Function` 对象。如果这是一个**直接调用**（如 `call @foo()`)，它会返回指向 `@foo` 的指针。如果这是一个**间接调用**（如通过函数指针 `call %ptr`)，它会返回 `nullptr`

```c++
llvm::PassPluginLibraryInfo getStaticCallCounterPluginInfo() {
  return { ...,
          [](PassBuilder &PB) {
            // 注册 #1: 手动调用 "print<static-cc>"
            PB.registerPipelineParsingCallback(...);
            
            // 注册 #2: 注册为分析服务
            PB.registerAnalysisRegistrationCallback(...);
          }};
};
```

**注册 #1 (`registerPipelineParsingCallback`)**: 注册 `print<static-cc>` 命令，让用户可以通过 `-passes="print<static-cc>"` 手动运行**打印 Pass**

**注册 #2 (`registerAnalysisRegistrationCallback`)**: 将 `StaticCallCounter` 注册为一个可用的**分析服务**。这是必不可少的一步，否则 `StaticCallCounterPrinter` 在调用 `MAM.getResult` 时会找不到这个服务而失败