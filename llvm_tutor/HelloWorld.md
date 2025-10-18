# HelloWorld Pass

## HelloWorld Pass使用——模块分析

```shell
llvm-tutor/build on  main [?] via 🅒 base 
➜ cmake -DLT_LLVM_INSTALL_DIR=~/projects/llvm-project/build ..
```

这个命令**并没有**将 `llvm-tutor` 的代码“插入”到 `llvm-project` 中。它的实际作用是：在 `llvm-tutor` 的 `build` 目录中，**配置一个独立的编译环境**，并告诉这个环境：“编译 `llvm-tutor` 所需要的所有依赖项（比如 LLVM 的头文件和库），请到 `-DLT_LLVM_INSTALL_DIR` 这个路径（也就是 `~/projects/llvm-project/build`）去找



```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ls
cmake_install.cmake CMakeFiles          Makefile

llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ make                                                         
[100%] Built target HelloWorld
```

在build环境中，不污染llvm-tutor源代码

编译后得到的`.dylib` 是 **动态链接库 (Dynamic Link Library)** 文件在苹果的 macOS 上的文件扩展名

它本身不能独立运行，但可以被 `opt` 这样的主程序加载，为主程序提供额外的功能（也就是 `HelloWorld` Pass)

```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ls -l ../lib/
total 7216
-rw-r--r--   1 manbin  staff    1389 10 17 20:52 cmake_install.cmake
drwxr-xr-x  15 manbin  staff     480 10 17 20:52 CMakeFiles
-rwxr-xr-x   1 manbin  staff  254696 10 17 20:52 libConvertFCmpEq.dylib
-rwxr-xr-x   1 manbin  staff  436328 10 17 20:52 libDuplicateBB.dylib
-rwxr-xr-x   1 manbin  staff  300248 10 17 20:52 libDynamicCallCounter.dylib
-rwxr-xr-x   1 manbin  staff  342480 10 17 20:52 libFindFCmpEq.dylib
-rwxr-xr-x   1 manbin  staff  167168 10 17 20:52 libHelloWorld.dylib
-rwxr-xr-x   1 manbin  staff  251992 10 17 20:52 libInjectFuncCall.dylib
-rwxr-xr-x   1 manbin  staff  218896 10 17 20:52 libMBAAdd.dylib
-rwxr-xr-x   1 manbin  staff  216416 10 17 20:52 libMBASub.dylib
-rwxr-xr-x   1 manbin  staff  299408 10 17 20:52 libMergeBB.dylib
-rwxr-xr-x   1 manbin  staff  373944 10 17 20:52 libOpcodeCounter.dylib
-rwxr-xr-x   1 manbin  staff  416032 10 17 20:52 libRIV.dylib
-rwxr-xr-x   1 manbin  staff  363160 10 17 20:52 libStaticCallCounter.dylib
-rw-r--r--   1 manbin  staff   21802 10 17 20:52 Makefile
```



clang编译.c文件得到中间IR

```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -O1 -S -emit-llvm ../../inputs/input_for_hello.c -o input_for_hello.ll
```



最后，用`libHelloWorld.dylib`作为插件，分析了`input_for_hello.ll`

```shell
llvm-tutor/build/HelloWorld on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ../lib/libHelloWorld.dylib -passes=hello-world -disable-output input_for_hello.ll
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
(llvm-tutor) Hello from: bar
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: fez
(llvm-tutor)   number of arguments: 3
(llvm-tutor) Hello from: main
(llvm-tutor)   number of arguments: 2
```

- `(llvm-tutor) Hello from: foo`

  `(llvm-tutor)   number of arguments: 1`

  这表示 Pass 找到了名为 `foo` 的函数，并检测到它有 1 个参数 (`int a`)。

- `(llvm-tutor) Hello from: bar`

  `(llvm-tutor)   number of arguments: 2`

  这表示 Pass 找到了名为 `bar` 的函数，并检测到它有 2 个参数 (`int a, int b`)。

- `(llvm-tutor) Hello from: fez`

  `(llvm-tutor)   number of arguments: 3`

  这表示 Pass 找到了名为 `fez` 的函数，并检测到它有 3 个参数 (`int a, int b, int c`)。

- `(llvm-tutor) Hello from: main`

  `(llvm-tutor)   number of arguments: 2`

  这表示 Pass 找到了名为 `main` 的函数，并检测到它有 2 个参数 (`int argc, char *argv[]`)。



## HelloWorld 源码

`llvm-tutor/HelloWorld.cpp`

```cpp
//=============================================================================
// FILE:
//    HelloWorld.cpp
//
// DESCRIPTION:
//    Visits all functions in a module, prints their names and the number of
//    arguments via stderr. Strictly speaking, this is an analysis pass (i.e.
//    the functions are not modified). However, in order to keep things simple
//    there's no 'print' method here (every analysis pass should implement it).
//
// USAGE:
//    New PM
//      opt -load-pass-plugin=libHelloWorld.dylib -passes="hello-world" `\`
//        -disable-output <input-llvm-file>
//
//
// License: MIT
//=============================================================================
```

文件开头给出该Pass描述：

该Pass为**analysis分析类** Pass （不修改输入）

**遍历一个模块中的所有函数，通过 stderr (标准错误输出) 打印它们的名称和参数数量**

`opt -load-pass-plugin=libHelloWorld.dylib -passes="hello-world" -disable-output <input-llvm-file>`使用该Pass



1. 引入必要头文件：

```cpp
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
```

- **`PassBuilder.h` 和 `PassPlugin.h`**: 这是编写基于 LLVM 新 Pass 管理器的插件所必需的头文件。它们提供了注册和管理 Pass 的核心功能。
- **`raw_ostream.h`**: 这个头文件提供了 `errs()`，一个用于向标准错误流（终端）打印信息的工具，类似于 C++ 中的 `std::cerr`。
- **`using namespace llvm;`**: 这是一个便利的声明，它允许我们直接使用 LLVM 命名空间下的类和函数（如 `Function`, `errs()`），而无需在每次使用时都加上 `llvm::` 前缀。**无需将 Pass 的内部实现暴露给外部世界 - 将所有内容都保留在一个匿名命名空间namespace中**



2. 核心逻辑 :

```c++
namespace {

// This method implements what the pass does
void visitor(Function &F) {
    errs() << "(llvm-tutor) Hello from: "<< F.getName() << "\n";
    errs() << "(llvm-tutor)   number of arguments: " << F.arg_size() << "\n";
}
  ...
}
```

定义一个名为visitor函数，接收一个Function对象引用作为入参，代表要分析的模块中的函数

对应地进行终端信息打印`errs() << ...`



3. 主体结构——定义Pass本身

```c++
// New PM implementation
struct HelloWorld : PassInfoMixin<HelloWorld> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    visitor(F);
    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};
```



`HelloWorld结构体`继承自 `PassInfoMixin<HelloWorld>`

新的 Pass 管理器定义依赖于多态，意味着并不存在显示的接口，所有的 Pass 是继承自 CRTP 模板`PassInfoMixin<PassT>`，其中需要有一个`run()`方法，接收一些 IR 单元和一个分析管理器，返回类型为 PreservedAnalyses

定义在`llvm/include/IR/PassManager.h `

```cpp
// A CRTP mix-in to automatically provide informational APIs needed for
// passes.
template <typename DerivedT> struct PassInfoMixin { ... };

// A CRTP mix-in that provides informational APIs needed for analysis passes.
template <typename DerivedT>
struct AnalysisInfoMixin : PassInfoMixin<DerivedT> { ... };
```

`PassInfoMixin`（Pass 信息混入）和 `AnalysisInfoMixin`（分析信息混入）是两个**辅助工具类**。当你编写一个 Pass 时，只需要让Pass 结构体继承自它们，就能自动获得一些便利的功能



**`PreservedAnalyses run(Function &F, ...)`**:   **接收要运行 Pass 的 IR 单元 (&F) 以及相应的 Pass 管理器**

这是 Pass 的**主入口函数**，返回一个 `PreservedAnalyses` 类型的对象

当 `opt` 工具运行这个 Pass 时，LLVM 的 Pass 管理器会遍历输入文件中的**每一个函数**，并为每个函数调用一次这个 `run` 方法

**`return PreservedAnalyses::all();`**:   

```c++
  static PreservedAnalyses all() {
    PreservedAnalyses PA;
    PA.PreservedIDs.insert(&AllAnalysesKey);
    return PA;
  }
```

```c++
SmallPtrSet<void *, 2> PreservedIDs;
```

创建 `PreservedAnalyses` 对象并插入 `AllAnalysesKey`（”全部有效“的标识符），存储在PA

 Pass在完成工作后，需要填写并提交给 Pass 管理器（PassManager）

告诉 LLVM 的 Pass 管理器：“**报告！我的任务完成了，并且我没有修改任何东西，之前的所有分析结果都完好无损，可以继续使用。**”



**`static bool isRequired() { return true; }`**:

这是一个非常重要的“开关”。当您使用 `clang -O0`（无优化）编译代码时，`clang` 会给每个函数加上一个 `optnone` 属性，告诉优化器跳过这个函数。

返回 `true` 意味着强制告诉 Pass 管理器：“无论有没有 `optnone` 属性，我的这个 Pass 都必须运行。” 这确保了我们的 Pass 在任何情况下都能被执行。



4. 向 LLVM 注册您的 Pass——将定义的HelloWorld结构体绑定在hello-world指令上

```c++
// New PM Registration
llvm::PassPluginLibraryInfo getHelloWorldPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "HelloWorld", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM, ...) {
                  if (Name == "hello-world") {
                    FPM.addPass(HelloWorld());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getHelloWorldPluginInfo();
}
```



Pass结尾部分——**`extern "C" ... llvmGetPassPluginInfo()`**: 这是 Pass 插件的**标准入口点**。当 `opt` 使用 `-load-pass-plugin` 加载您的 `.dylib` 文件时，它会寻找并调用这个函数



**`getHelloWorldPluginInfo()`**: 这个函数返回一个 `PassPluginLibraryInfo` 结构体，其中包含了插件的元信息

**`PB.registerPipelineParsingCallback(...)`**: 这是最关键的注册逻辑。它告诉 LLVM 的 `PassBuilder`：

- “请注册一个回调函数。当用户在 `-passes=` 参数中提供一个名字时，就调用这个回调。”
- **`if (Name == "hello-world")`**: 在回调函数内部，我们检查用户提供的名字是不是 `hello-world`。
- **`FPM.addPass(HelloWorld());`**: 如果名字匹配，我们就创建一个 `HelloWorld` Pass 的实例，并将其添加到函数 Pass 管理器 (`FunctionPassManager`) 中。
- **`return true;`**: 告诉 `PassBuilder` 我们已经成功处理了这个名字