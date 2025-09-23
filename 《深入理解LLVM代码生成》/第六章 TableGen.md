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