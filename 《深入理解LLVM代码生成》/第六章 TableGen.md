# 第六章 TableGen

每一款硬件都需要编译器实现相应的高级语言转换，此时需要了解目标机器的指令集信息（支持哪些指令，指令有什么属性、使用什么寄存器、指令间存在什么依赖）。虽然硬件指令集都不相同，**但都包含指令、寄存器、调用约定等信息**，所以可以进行抽象后端实现为两层：**具体硬件信息**、**硬件无关的编译器框架**

**不同目标硬件都包含了丰富的指令或寄存器信息，直接描述这些信息会非常复杂且难做到统一**，因此设计**后端信息描述语言即目标描述语言TableGen**

## 6.1 目标描述语言（TableGen的词法语法）

6.1.1 在词法分析阶段对TD文件进行分析得到对应token（单词符号）

1. 数值字面量文法
2. 字符串字面量文法：单行`“   ”` 且中间不能有`"`或转义字符`\`     跨行`[{   }]`且中间不能有`}]`	
3. 关键字——描述类型、语法结构等信息
4. 标识符——定义了变量,考虑到了别名`$`
5. 特殊运算符——TableGen特有的内置处理方式，如`!`运算符，`!add`表示对多个操作数进行求和计算
6. 基本分隔符
7. 其他语法——提供`include语法`,引入前实现预处理



6.1.2 具有丰富的语法规则

dag：表示可嵌套的DAG，节点中的运算符必须是一个实例化的记录，**DAG大小等于参数的数量**

`operator arg1, arg2, arg3, ...`一个操作数显示定义且必须是一个记录，0个或多个参数



**后缀值**是在**简单值后加约束**，如`0b110`表示的就是二进制值`0b0110`

**复合值**：通过`#`进行组合，如`let str = "12" # "ab"`表示`12ab`赋值给`str`

 

**记录：TableGen最主要的目的之一就是生成记录，然后后端基于记录进行分析得到最终结果**

记录可以被看作是**有名字、有类型、具有特定属性的结构体**

TableGen分别通过`def`和`class`定义记录，并提供能批量定义记录的高级语法`mutliclass` 和`defm`

```c++
def record_example {
    int a=1;
    string b="def example";
}
```

```c++
// 使用class定义一个记录类，然后使用def进行实例化
class TestInst{
    string asmname;
    bits<32> encoding;
}
def ADD: TestInst{
    let asmname="add";
    let encoding{31-26}=1;
}
def MUL: TestInst{
    let asmname="mul";
    let encoding{31-26}=2;
}
```



用`mutliclass` 和`defm`定义一组记录:

```cpp
// 实现批量处理
class Instr<bits<4> op, string desc> {
    bits<4> opcode = op;
    string name = desc;
}
multiclass RegInstr {
    def rr : Instr<0b1111,"rr">;
    def rm : Instr<0b0000,"rm">;
}

defm MyBackend_:RegInstr;
```

`multiclass`批量定义了两个记录类如下：

```c++
class rr {
    bits<4> opcode = 0b1111;
    string nameDes = "rr";
}
class rm {
    bits<4> opcode = 0b0000;
    string nameDes = "rm";
}
```

然后通过def进行实例化，实例化对象分别是`MyBackend_rm`和`MyBackend_rr`

使用llvm-tblgen编译得到相关记录如下：

```cpp
------------- Classes -----------------
class Instr<bits<4> Instr:op = { ?, ?, ?, ? }, string Instr:desc = ?> {
  bits<4> opcode = { Instr:op{3}, Instr:op{2}, Instr:op{1}, Instr:op{0} };
  string name = Instr:desc;
}
------------- Defs -----------------
def MyBackend_rm {     // RegInstr
  bits<4> opcode = { 0, 0, 0, 0 };
  string nameDes = "rm";
}
def MyBackend_rr {     // RegInstr
  bits<4> opcode = { 1, 1, 1, 1 };
  string nameDes = "rr";
}
```





## 6.2 TableGen工具链

LLVM提供的工具**llvm-tblgen可将TD文件转换成和LLVM框架配合的C++代码**：

llvm-tblgen将TD文件转换成记录，即**工具链前端**

对记录中信息进行抓取、分析和组合以生成inc头文件，通常和LLVM框架配合使用，即**工具链后端**



后端由开发者定义，一般有以下3种：LLVM后端，Clang后端，通用后端

其中**Clang属于解析记录生成与架构无关的一些信息**，如语法树信息，诊断信息以及类属性等

而**LLVM则是解析记录生成与架构有关的一些信息**，如描述架构寄存器和指令信息的头文件，或用于指导代码生成、指令选择的代码片段

通用后端则是不进行记录解析，而是进行简单的处理



6.2.1 从TD定义到记录

由于LLVM支持多种后端，所以设计时将记录类分为两种：适用于所有后端的基类记录类、适用于某一后端的派生记录类

```shell
llvm-tblgen -I your_dir/llvm-project/llvm/include/ -I your_dir/llvm-project/llvm/lib/Target/BPF --print-records your_dir/llvm-project/llvm/lib/Target/BPF/BPF.td
```

**将TD文件转成记录**



6.2.2 从记录到C++代码

通过引入`-gen-dag-isel`参数，使用`llvm-tblgen`**生成专门头文件来保存相关信息，.inc文件最为重要一部分内容时指令匹配表**`MatcherTable`





## 6.3 扩展阅读：如何在TD文件中定义匹配

**6.3.1 隐式定义匹配**

**一句话解释**：让编译器知道一条指令除了“看得见”的输入输出外，还有哪些“看不见”的副作用，最典型的就是**条件码寄存器**；在定义指令时，**将这个隐式的寄存器明确地加入到输入或输出列表中**

像`sub`（减法）这样的指令，除了计算出结果，还会**隐式地更新**这个状态寄存器中的标志位（如零标志位ZF、进位标志位CF）。而紧随其后的条件跳转指令（如`jz` - 如果为零则跳转）则会**隐式地读取**这些标志位来做决策。如果我们不告诉编译器这种隐藏的依赖关系，它可能会错误地在`sub`和`jz`指令之间插入其他无关指令，破坏了状态寄存器中的值，导致程序出错



**6.3.2 复杂匹配模板**

**一句话解释**：为可重用的、复杂的“子模式”命名，专门用于解决**内存寻址模式**的匹配问题

像x86这样的CISC架构拥有非常强大的内存寻址能力，例如 `[Base + Index*Scale + Displacement]`，如果我们为每一条能使用这种寻址模式的指令（`MOV`, `ADD`, `SUB`...）都重复书写一遍对应的DAG匹配模式，`.td`文件将会变得极其冗长且难以维护

`ComplexPattern`允许我们将这个复杂的地址计算模式提取出来，并给它起一个名字（例如`addr`），然后在所有需要它的地方重复使用这个名字即可

```c++
// 定义一个名为 `addr` 的复杂模式
// 它匹配的结果是一个32位地址 (i32)
// 它会捕获2个操作数 (基址寄存器和偏移量)
// 匹配成功后，会调用C++函数 "SelectAddr" 来处理
def addr : ComplexPattern<i32, 2, "SelectAddr",
    [
        // 模式1: [reg + imm]
        (add GPR32:$base, i32imm:$offset),
        // 模式2: [imm + reg] (交换律)
        (add i32imm:$offset, GPR32:$base),
        // 模式3: [reg] (偏移量为0)
        (GPR32:$base)
    ]
>;
```



**6.3.3 匹配规则支持类**

**一句话解释**：它们是模式匹配中的“瑞士军刀”，提供了一些可重用的“小工具”，如`PatFrag`和`SDNodeXForm`，让模式定义更灵活、更强大 





**6.3.4 总结**

**隐式定义**处理指令的副作用，**`ComplexPattern`处理复杂的寻址，而支持类**则提供了代码重用和自定义检查的能力，它们共同确保了LLVM能够精确、高效地为各种现实世界的复杂CPU生成代码





## 6.4 面试题 Q&A 

**Q1: 既然你自己完整手写过编译器后端（如 Sysy_rvcp 项目），为什么像 LLVM 这样的工业级编译器还要专门发明 TableGen 这种语言来描述目标机器？**

 **A:** 在实现精简指令集或教学级别的后端时，直接用 C++ 硬编码（如手写汇编生成或图节点降级）是够用且直观的。但 LLVM 作为一个工业级框架，需要支持成百上千种微架构及海量的指令变体。 如果全用 C++ 写，会产生海量极难维护的样板代码。TableGen 的核心价值在于**“数据与逻辑分离”**。通过它提供的 `class` 继承和 `multiclass` 等特性，开发者可以一次性批量实例化出一条指令的多种变体（如寄存器版、立即数版），极大地提高了后端开发效率和代码复用率，让复杂的指令集管理变得高度结构化。



**Q2: TableGen（即 `.td` 文件）是如何最终参与到 LLVM 的编译工作中的？它是在编译器运行时解析的吗？**

 **A:** 不是运行时的解析工具，TableGen 的工作完全在**构建期（Build Time）**完成。 在真正编译 LLVM 自身源码之前，构建系统会先调用 `llvm-tblgen` 工具。该工具读取 `.td` 文件，在内存中完成解析，然后通过各种代码生成器（Emitters）输出纯 C++ 的头文件（通常是 `.inc` 格式的宏、枚举或数组）。当系统使用 C++ 编译器（如 GCC/Clang）编译 LLVM 后端源码时，这些 `.inc` 文件会被直接 `#include` 进去。因此，最终编译出的 LLVM 编译器在运行（编译用户代码）时，只执行纯粹的 C++ 逻辑，完全不需要再去解析 `.td` 文件。



**Q3: 在调试 LLVM 后端时，如果发现输出的某条 RISC-V 汇编指令格式不对，你应该去修改后端的 C++ 代码还是修改 `.td` 文件？** 

**A:** 绝大多数情况下应该修改 **`.td` 文件**。 因为汇编的打印格式（AsmString）、机器码的二进制编码规则以及寄存器操作数的约束，都是在 TableGen 中静态定义并由工具自动生成的。只有当这条指令涉及非常特殊且复杂的自定义展开逻辑（Custom Lowering），或者复杂的伪指令展开时，才需要去修改后端的 C++ 源文件。



**Q4: 如果你写了一段 TableGen 逻辑，想验证生成的指令信息对不对，但又不想花费几十分钟去重新编译整个 LLVM，有什么高效的调试方法？**

 **A:** 可以直接使用 `llvm-tblgen` 工具自带的**通用后端（调试功能）**。 在命令行执行 `llvm-tblgen -print-records RISCV.td`（并带上相关的 include 路径）。这个命令会跳过 C++ `.inc` 文件的生成阶段，直接将词法和语法分析后展开的所有终态记录（Records）以人类可读的纯文本或 JSON 格式打印在终端上。这是一种非常高效的静态排错手段，可以立刻验证我的实例化逻辑是否正确。



**Q5: 在 TableGen 中经常能看到定义为 `dag` 类型的 Pattern（模式匹配），它在后端的哪个阶段发挥作用？底层是如何高效实现的？**

 **A:** 它主要在**指令选择（Instruction Selection）**阶段发挥作用。 LLVM 并没有把这些 `dag` 模式生硬地转换成庞大的 `switch-case` 嵌套 C++ 代码（那会导致编译极慢且体积臃肿）。相反，TableGen 会将这些模式编译成一个极其紧凑的常量字节码数组，称为 **`MatcherTable`**。在编译器运行时，后端的指令选择器（DAGISel）相当于一个基于状态机的轻量级解释器，它读取这套字节码，将中端传来的 LLVM IR 节点高效地匹配并替换为特定目标机器的物理指令节点。



**Q6: 从指令模式匹配的角度看，RISC-V 架构和传统的 x86 架构在 TableGen 定义上有什么显著的区别？**

 **A:** 最大的区别在于对**隐式状态（Implicit Condition Codes）**的依赖。 x86 的传统指令（如算术运算）通常会隐式更新状态标志寄存器（EFLAGS），而后续的条件跳转指令会隐式读取它。因此在 x86 的 TableGen 定义中，必须显式声明这些副作用（Defs/Uses），否则会影响指令调度的正确性。 而 RISC-V 的基础整数指令集秉持精简原则，刻意去除了条件码寄存器，采用了融合的比较跳转（如 `beq` 指令），这就消除了隐藏的状态依赖，使得 RISC-V 的调度图和 TableGen 模式匹配更加清晰、干净。