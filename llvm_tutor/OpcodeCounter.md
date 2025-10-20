# OpcodeCounter

## OpcodeCounter Pass使用 —— 操作码计数

OpcodeCounter 是一个分析过程，用于打印 **LLVM IR 操作码**的摘要 在输入模块的每个函数中都会遇到。此过程可以使用预定义的优化管道之一**自动运行**

举例如图：

```
=================================================
LLVM-TUTOR: OpcodeCounter results for `main`
=================================================
OPCODE               #N TIMES USED
-------------------------------------------------
load                 2
br                   4
icmp                 1
add                  1
ret                  1
alloca               2
store                4
call                 4
-------------------------------------------------
```

```c++
/// llvm-project/llvm/lib/IR/Instruction.cpp
const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case IndirectBr: return "indirectbr";
  case Invoke: return "invoke";
  case Resume: return "resume";
  case Unreachable: return "unreachable";
  case CleanupRet: return "cleanupret";
  case CatchRet: return "catchret";
  case CatchPad: return "catchpad";
  case CatchSwitch: return "catchswitch";
  case CallBr: return "callbr";
	...
	}
```



```sh
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/clang -emit-llvm -c ../inputs/input_for_cc.c -o input_for_cc.bc
llvm-tutor/build on  main [?] via 🅒 base 
➜ ls
bin                 cmake_install.cmake CMakeCache.txt      CMakeFiles          HelloWorld          input_for_cc.bc     lib                 Makefile            test                tools      
```

**使用 `clang` 生成 IR 文件** ，将 C 语言源文件 编译成 bitcode



```c++
llvm-tutor/build on  main [?] via 🅒 base 
➜ ~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libOpcodeCounter.dylib --passes="print<opcode-counter>" -disable-output input_for_cc.bc
  
Printing analysis 'OpcodeCounter Pass' for function 'foo':
=================================================
LLVM-TUTOR: OpcodeCounter results
=================================================
OPCODE               #TIMES USED
-------------------------------------------------
ret                  1         
-------------------------------------------------

Printing analysis 'OpcodeCounter Pass' for function 'bar':
=================================================
LLVM-TUTOR: OpcodeCounter results
=================================================
OPCODE               #TIMES USED
-------------------------------------------------
call                 1         
ret                  1         
-------------------------------------------------

Printing analysis 'OpcodeCounter Pass' for function 'fez':
=================================================
LLVM-TUTOR: OpcodeCounter results
=================================================
OPCODE               #TIMES USED
-------------------------------------------------
call                 1         
ret                  1         
-------------------------------------------------

Printing analysis 'OpcodeCounter Pass' for function 'main':
=================================================
LLVM-TUTOR: OpcodeCounter results
=================================================
OPCODE               #TIMES USED
-------------------------------------------------
add                  1         
call                 4         
ret                  1         
load                 2         
br                   4         
alloca               2         
store                4         
icmp                 1         
-------------------------------------------------

```

使用 `opt` 工具来加载 `OpcodeCounter` 插件并分析刚刚生成的 `.bc` 文件

**`--passes="print<opcode-counter>"`**

关键！告诉 `opt` 运行 `OpcodeCounter Pass`  的**打印版本**。因为 `OpcodeCounter` 是一个分析 Pass，它本身只计算结果而不打印



**使用优化管道将Pass自动注册到流水线中**

通过简单地指定优化级别来运行 **OpcodeCounter** （例如 `-O{1|2|3|s}` ）

```shell
~/projects/llvm-project/build/bin/opt -load-pass-plugin ./lib/libOpcodeCounter.dylib --passes='default<O1>' -disable-output input_for_cc.bc
```

在 `-O1` 流水线进行到**矢量化**阶段时运行，最终同样可以获得相同的结果



## OpcodeCounter 源码

### .h

```cpp
/// OpcodeCounter.h
#ifndef LLVM_TUTOR_OPCODECOUNTER_H
#define LLVM_TUTOR_OPCODECOUNTER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

// 用StringMap来存储：操作码的名字(Key)->出现的次数(Value)
using ResultOpcodeCounter = llvm::StringMap<unsigned>; // 类型别名

// 继承AnalysisInfoMixin来定义一个分析Pass
struct OpcodeCounter : public llvm::AnalysisInfoMixin<OpcodeCounter> {
  using Result = ResultOpcodeCounter;
  Result run(llvm::Function &F, ...);
  OpcodeCounter::Result generateOpcodeMap(llvm::Function &F);
  
  static bool isRequired() { return true; }

private:
  static llvm::AnalysisKey Key;
  friend struct llvm::AnalysisInfoDixin<OpcodeCounter>;
};
```

| static llvm::AnalysisKey Key; | 类似于 ID, 可以认证这个特殊的 analysis pass 类 |
| ----------------------------- | ---------------------------------------------- |

这个 Key 要在 cpp 文件中初始化，`AnalysisManager` 就是通过这个唯一的静态 `Key` 来识别和缓存 `OpcodeCounter` 的分析结果的



**相关类**

**AnalysisInfoMixin**

```c++
 
/// 继承了PassInfoMixin， 使用 CRTP 混合技术
/// 为 analysis pass 提供相比普通 pass 额外的必要的 API: ID()
template <typename DerivedT>
struct AnalysisInfoMixin : PassInfoMixin<DerivedT> {
  /// 为该 analysis type 提供独立的 ID
  /// 该 ID 是一个指针类型
  ///  要求子类提供一个静态的 AnalysisKey 名为 Key
 
  static AnalysisKey *ID() {
    static_assert(std::is_base_of<AnalysisInfoMixin, DerivedT>::value,
                  "Must pass the derived type as the template argument!");
    return &DerivedT::Key;
  }
};
```



```c++
//------------------------------------------------------------------------------
// New PM interface for the printer pass
// 这个 Pass 的唯一职责就是打印 OpcodeCounter 的分析结果
// 这是一个很好的设计模式，将计算和展示分离开
//------------------------------------------------------------------------------
class OpcodeCounterPrinter : public llvm::PassInfoMixin<OpcodeCounterPrinter> {
public:
  explicit OpcodeCounterPrinter(llvm::raw_ostream &OutS) : OS(OutS) {} //llvm::raw_ostream &OS;对外输入流的引用
  llvm::PreservedAnalyses run(llvm::Function &Func,
                              llvm::FunctionAnalysisManager &FAM);
  static bool isRequired() { return true; }

private:
  llvm::raw_ostream &OS;
};
#endif
```

**`explicit OpcodeCounterPrinter(llvm::raw_ostream &OutS)`**: 它的构造函数接收一个输出流（`raw_ostream`）作为参数，比如 `llvm::errs()`，这样它就知道要把结果打印到哪里

`explicit` 禁止编译器进行“隐式类型转换”

`: OS(OutS)` 是**构造函数**的一部分，专门用来在对象创建时**初始化其成员变量**

即：

```c++
private:
  llvm::raw_ostream &OS; // 这个成员变量——OS,传递给成员变量OS 这个构造函数的入参 OutS，即做了一个成员变量的初始化
```





### .cpp

```c++
//    遍历一个函数中的所有指令，并统计每一种 LLVM IR 操作码被使用了多少次
//    将输出打印到 stderr (标准错误输出)

// 用法:
//    1. 新版 PM
//      opt -load-pass-plugin libOpcodeCounter.dylib -passes="print<opcode-counter>" `\`
//        -disable-output <输入的llvm文件>
//    2. 通过优化管线自动运行 - 新版 PM
//      opt -load-pass-plugin libOpcodeCounter.dylib --passes='default<O1>' `\`
//        -disable-output <输入的llvm文件>
```

```c++
OpcodeCounter::Result OpcodeCounter::generateOpcodeMap(llvm::Function &Func) {
  OpcodeCounter::Result OpcodeMap;

  for (auto &BB : Func) {      // 遍历函数中的每一个基本块 (Basic Block)
    for (auto &Inst : BB) {    // 遍历基本块中的每一条指令 (Instruction)
      StringRef Name = Inst.getOpcodeName(); // 获取指令的操作码名称

      if (OpcodeMap.find(Name) == OpcodeMap.end()) { // 如果是第一次遇到
        OpcodeMap[Name] = 1;                         // 初始化计数为 1
      } else {
        OpcodeMap[Name]++;                           // 否则，计数加 1
      }
    }
  }

  return OpcodeMap;
}
```

```c++
// OpcodeCounter 的 run 方法
OpcodeCounter::Result OpcodeCounter::run(llvm::Function &Func, ...) {
  return generateOpcodeMap(Func);
}

// OpcodeCounterPrinter 的 run 方法
PreservedAnalyses OpcodeCounterPrinter::run(Function &Func,
                                            FunctionAnalysisManager &FAM) {
  // 从 FAM 获取 OpcodeCounter 的分析结果
  auto &OpcodeMap = FAM.getResult<OpcodeCounter>(Func);

  // ... (打印表头) ...

  // 调用辅助函数打印结果
  printOpcodeCounterResult(OS, OpcodeMap);
  return PreservedAnalyses::all();
}
```

**`FAM.getResult<OpcodeCounter>(Func)`**: 它向函数分析管理器（`FunctionAnalysisManager`）请求 `OpcodeCounter` 对当前函数 `Func` 的分析结果。`AnalysisManager` 会自动检查缓存或按需运行 `OpcodeCounter`，然后返回结果

拿到结果后，它就调用 `printOpcodeCounterResult` 辅助函数将结果以格式化的表格输出，因此在文件前面应该先对 `printOpcodeCounterResult` 进行声明



```c++
llvm::PassPluginLibraryInfo getOpcodeCounterPluginInfo() {
  return { ...,
        [](PassBuilder &PB) {
          // #1. 注册用于 "-passes=print<opcode-counter>"
          PB.registerPipelineParsingCallback(...);

          // #2. 注册用于 "-O1" 等优化级别
          PB.registerVectorizerStartEPCallback(...);
          
          // #3. 注册 OpcodeCounter 作为一个分析服务
          PB.registerAnalysisRegistrationCallback(...);
          }
        };
}
```



动态库（`.dylib` ）与 `opt` 等 LLVM 工具之间的**唯一连接点**

```c++
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getOpcodeCounterPluginInfo(); // 上面的注册函数
}
```

**`extern "C"` 的目的**：保证函数名在编译后不被 C++ 编译器修改，使得 `opt` 能够通过一个固定的、标准的名字 (`llvmGetPassPluginInfo`) 找到它



```c++
//------------------------------------------------------------------------------
// Helper functions - implementation
//------------------------------------------------------------------------------
static void printOpcodeCounterResult(raw_ostream &OutS,
                                     const ResultOpcodeCounter &OpcodeMap) {
  OutS << "================================================="
               << "\n";
  OutS << "LLVM-TUTOR: OpcodeCounter results\n";
  OutS << "=================================================\n";
  const char *str1 = "OPCODE";
  const char *str2 = "#TIMES USED";
  OutS << format("%-20s %-10s\n", str1, str2);
  OutS << "-------------------------------------------------"
               << "\n";
  for (auto &Inst : OpcodeMap) {
    OutS << format("%-20s %-10lu\n", Inst.first().str().c_str(),
                           Inst.second);
  }
  OutS << "-------------------------------------------------"
               << "\n\n";
}
```

之前声明的打印辅助函数，OutS对应的格式化输出

`OutS` **最终都指向了一个输出流对象**（ `llvm::errs()`）