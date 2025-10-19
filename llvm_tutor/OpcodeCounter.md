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