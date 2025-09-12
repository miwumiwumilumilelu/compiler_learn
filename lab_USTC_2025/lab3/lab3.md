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



### 1.1 初识龙芯汇编和GNU汇编伪指令

#### 1.1.1 浮点数的 **IEEE 754 浮点数标准**

`ffint.s.w $fd, $fj`

- 行为：选择**浮点寄存器** `$fj` 中的整数型定点数转换为单精度浮点数，得到的单精度浮点数写入到浮点寄存器 `$fd` 中

  

在执行下面的指令前，`$ft0` 寄存器的值为 `0x0000_0000_0000_0008`

`ffint.s.w $ft1, $ft0`

执行后，`$ft1` 寄存器的值为?



根据 **IEEE 754 浮点数标准**，一个 32 位单精度浮点数由三部分组成：

* **符号位**（1 位）：表示正负。

* **指数**（8 位）：表示数字的大小范围。

* **尾数**（23 位）：表示数字的精度。



**步骤：**

**a) 标准化**

首先，我们需要将 `8` 转换为科学记数法形式，但基数是 2： **8=1.0×2^3   即0b1000**

**b) 确定每个部分的值**

- **符号位**：因为 8 是正数，所以符号位是 **`0`**。
- **指数**：科学记数法中的指数是 `3`。在 IEEE 754 标准中，指数需要加上一个固定的偏移量 **127**（这个偏移量是为了让指数既能表示正数也能表示负数）。
  - 指数值 = 3 + 127 = 130
  - 将 130 转换为 8 位二进制：`1000 0010`
- **尾数**：标准化后的数字是 `1.0`。尾数只记录小数点后的部分。
  - `1.0` 的尾数部分是 `0`。
  - 为了凑足 23 位，我们需要在后面补 23 个 `0`：
  - `00000000000000000000000`

**c) 组合结果**

现在，我们将这三部分按顺序拼接起来：

| 符号位 (1 bit) | 指数 (8 bits)  | 尾数 (23 bits)                |
| -------------- | -------------- | ----------------------------- |
| **`0`**        | **`10000010`** | **`00000000000000000000000`** |

**组合在一起得到完整的 32 位二进制数：**`0100 0001 0000 0000 0000 0000 0000 0000`

**d) 四位一组转成0x**

- `0100` -> `4`
- `0001` -> `1`
- `0000` -> `0`
- `0000` -> `0`



执行后，`$ft1` 寄存器的值为 `0x0000_0000_4100_0000`（`8.0` 的 FP32 表示）



#### 1.1.2 LA64 汇编样例



##### **一、主要学习其栈帧分配思想**



```c
int globalInt;
float globalFloat;
int globalIntArray[10];

int main(void) {
    return 0;
}
```

编译器生成的汇编代码如下，其中有大量的伪指令，注释中有对各条伪指令的说明:

```c
# Global variables
    .text                         # 后面的内容将放在代码段 (但是由于 .section, 全局变量实际被放到了 BSS 段)
    .section .bss, "aw", @nobits  # 将后面的全局变量放到 BSS 段, 并设置属性

    .globl globalInt              # 标记 globalInt 为全局符号
    .type globalInt, @object      # 标记 globalInt 为数据对象/变量
    .size globalInt, 4            # 标记 globalInt 的大小为 4 字节
globalInt:
    .space 4                      # 为 globalInt 分配 4 字节空间

    .globl globalFloat            # 标记 globalFloat 为全局符号
    .type globalFloat, @object    # 标记 globalFloat 为数据对象/变量
    .size globalFloat, 4          # 标记 globalFloat 的大小为 4 字节
globalFloat:
    .space 4                      # 为 globalFloat 分配 4 字节空间

    .globl globalIntArray         # 标记 globalIntArray 为全局符号
    .type globalIntArray, @object # 标记 globalIntArray 为数据对象/变量
    .size globalIntArray, 40      # 标记 globalIntArray 的大小为 40 字节 (4 * 10)
globalIntArray:
    .space 40                     # 为 globalIntArray 分配 40 字节空间

# Functions
    .text                 # 后面的内容将放在代码段
    .globl main           # 标记 main 为全局符号, main 函数是程序的入口, 因此这个标记是必须的
    .type main, @function # 标记 main 为函数
main:
    st.d $ra, $sp, -8
    st.d $fp, $sp, -16
    addi.d $fp, $sp, 0
    addi.d $sp, $sp, -16
.main_label_entry:
# ret i32 0
    addi.w $a0, $zero, 0
    b main_exit
main_exit:
    addi.d $sp, $sp, 16
    ld.d $ra, $sp, -8
    ld.d $fp, $sp, -16
    jr $ra
```



`main`函数部分：

* **函数序言：在执行前设置好栈帧**

  ```c
  main:
      st.d $ra, $sp, -8
      st.d $fp, $sp, -16
      addi.d $fp, $sp, 0
      addi.d $sp, $sp, -16	
  ```

  

  **流程：**

  假设

  **`$ra` 寄存器**：返回地址寄存器 `$ra` 中保存着 `main` 函数执行完后，程序应该返回去执行的下一条指令的地址（例如 `0xBADBEEF`）。

  **`$fp` 寄存器**：帧指针 `$fp` 中保存着调用者的帧指针（例如 `0x9000`）。

  ```c
  高地址 ^
          |
          +-------------------+ <--- 0x1010 （假设调用者栈帧的某个位置）
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1008
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1000  <--- $sp (初始值)
          |   (未使用的区域)  |
          +-------------------+
          |        ...        |
          +-------------------+
  低地址 v
  ```

  * `st.d $ra, $sp, -8`：将返回地址寄存器 **`$ra`** 的值存储到栈上。当 `main` 函数执行完毕后，程序需要知道回到哪里继续执行（例如回到启动代码），这个返回地址就保存在 `$ra` 中。

  ```c
  高地址 ^
          |
          +-------------------+ <--- 0x1010
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1008
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1000  <--- $sp (初始值)
          |     $ra 备份      |  (0xBADBEEF)
          +-------------------+ <--- 0x0FF8
          |   (未使用的区域)  |
          +-------------------+
          |        ...        |
  低地址 v
  ```

  * `st.d $fp, $sp, -16`：将帧指针寄存器 **`$fp`** 的值存储到栈上。帧指针用于定位栈上的局部变量和参数，保存它以便在函数返回时恢复。

  ```c
  高地址 ^
          |
          +-------------------+ <--- 0x1010
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1008
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1000  <--- $sp (初始值)
          |     $ra 备份      |  (0xBADBEEF)
          +-------------------+ <--- 0x0FF8
          |     $fp 备份      |  (0x9000)
          +-------------------+ <--- 0x0FF0
          |   (未使用的区域)  |
          +-------------------+
          |        ...        |
  低地址 v
  ```

  * `addi.d $fp, $sp, 0`：将当前栈指针 **`$sp`** 的值复制给帧指针 **`$fp`**，以作为栈帧基地址建立新的栈帧。

  * `addi.d $sp, $sp, -16`：分配栈空间。`$sp` 减去 16 字节，为局部变量和保存的寄存器分配空间。`-16` 是因为之前保存了 `$ra` 和 `$fp`，每个占用 8 字节。

  ```c
  高地址 ^
          |
          +-------------------+ <--- 0x1010
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1008
          |   调用者栈帧数据  |
          +-------------------+ <--- 0x1000  <--- $fp (新栈帧的基址)
          |     $ra 备份      |  (0xBADBEEF)
          +-------------------+ <--- 0x0FF8  （相对于$fp是 $fp-8）
          |     $fp 备份      |  (0x9000)
          +-------------------+ <--- 0x0FF0  <--- $sp (新栈帧的顶部)
          |   局部变量/其它   |  (分配了16字节空间，此时这里是 $sp$)
          +-------------------+ <--- 0x0FE8  （相对于$fp是 $fp-24）
          |        ...        |
  低地址 v
  ```

  

* **函数完成退出：**

  ```c
  main_exit:
      addi.d $sp, $sp, 16
      ld.d $ra, $sp, -8
      ld.d $fp, $sp, -16
      jr $ra
  ```

  

##### 二、示例1：返回值问题

```c
源程序：
int main(void) {
    return 0;
}

汇编代码：
    .text                 # 标记代码段
    .globl main           # 标记 main 为全局符号
    .type main, @function # 标记 main 为函数
main:
    st.d $ra, $sp, -8     # 保存返回地址
    st.d $fp, $sp, -16    # 保存调用者的栈帧指针
    addi.d $fp, $sp, 0    # 设置新的栈帧指针
    addi.d $sp, $sp, -16  # 入栈, 为栈帧分配 16 字节的空间

    addi.w $a0, $zero, 0  # 将返回值设置成 0

    addi.d $sp, $sp, 16   # 出栈, 恢复原来的栈指针
    ld.d $ra, $sp, -8     # 恢复返回地址
    ld.d $fp, $sp, -16    # 恢复栈帧指针
    jr $ra                # 返回至调用者
```



##### 三、示例2：注意比较对象是寄存器还是立即数

```c
源程序：
int a;
int main(void) {
    a = 4;
    if (a > 3) {
        return 1;
    }
    return 0;
}


汇编代码：
    .text                        # 标记代码段
    .section .bss, "aw", @nobits # 将后面的全局变量放到 BSS 段, 并设置属性
    .globl a                     # 标记 a 为全局符号
    .type a, @object             # 标记 a 为数据对象/变量
    .size a, 4                   # 标记 a 的大小为 4 字节
a:
    .space 4                     # 为 a 分配 4 字节的空间

    .text                    # 标记代码段
    .globl  main             # 标记 main 为全局符号
    .type   main, @function  # 标记 main 为函数
main: # 进入 main 函数
    st.d $ra, $sp, -8        # 保存返回地址
    st.d $fp, $sp, -16       # 保存调用者的栈帧指针
    addi.d $fp, $sp, 0       # 设置新的栈帧指针
    addi.d $sp, $sp, -16     # 入栈, 为栈帧分配 16 字节空间
.main_label_entry: # 分支判断入口
    addi.w  $t4, $zero, 4    # t4 = 4
    addi.w  $t2, $zero, 3    # t2 = 3
    la.local $t0, a          # 将 a 所处的内存地址加载入 t0
    st.w $t4, $t0, 0         # 将 t4 的数据保存入 t0 指向的地址中
    ld.w $t1, $t0, 0         # $t1 = a 的值
    blt $t2, $t1, .main_then # 将 t2 和 t1 比较,如果 t2 < t1 则跳转到 main_then
    b   .main_else           # 否则跳转到 .main_else
.main_else:
    addi.w  $a0, $zero, 0    # 设置返回值为 0
    b   .main_label_exit     # 跳转到 .main_label_exit
.main_then:
    addi.w  $a0, $zero, 1    # 设置返回值为 1
    b   .main_label_exit     # 跳转到 .main_label_exit
.main_label_return: # 退出 main 函数
    addi.d $sp, $sp, 16      # 出栈, 恢复原来的栈指针
    ld.d $ra, $sp, -8        # 恢复返回地址
    ld.d $fp, $sp, -16       # 恢复栈帧指针
    jr $ra                   # 返回至调用者
```

`la.local $t0 ,a`的作用就是**构建全局变量 `a` 的完整 64 位内存地址**，并将其存入一个通用寄存器 `$t0`

**注意立即数比较还是寄存器取值比较：如**

```c
addi.w  $t4, $zero, 4    # t4 = 4
addi.w  $t2, $zero, 3    # t2 = 3
blt $t2, $t4, .main_then
    
    立即数比较，不符合源代码↑
    
addi.w  $t4, $zero, 4    # t4 = 4
addi.w  $t2, $zero, 3    # t2 = 3
la.local $t0, a          # 将 a 所处的内存地址加载入 t0
st.w $t4, $t0, 0         # 将 t4 的数据保存入 t0 指向的地址中
ld.w $t1, $t0, 0         # $t1 = a 的值
blt $t2, $t1, .main_then
    
    a < 3 符合源代码，取a中的值需要先知道a的内存地址$t0，然后加载ld.w↑
```



##### 四、示例3：注意main返回值，可能需要修改a0

```c
源程序：
int add(int a, int b) {
    return a + b;
}

int main(void) {
    int a;
    int b;
    int c;
    a = 1;
    b = 2;
    c = add(a, b);
    output(c);
    return 0;
}



汇编程序：
    .text                   # 标记代码段
    .globl  add             # 标记 add 全局可见（必需）
    .type   add, @function  # 标记 main 是一个函数
add:
    st.d $ra, $sp, -8       # 保存返回地址, 在这里即为 bl add 指令的下一条指令地址
    st.d $fp, $sp, -16      # 保存调用者的栈帧指针
    addi.d $fp, $sp, 0      # 设置新的栈帧指针
    addi.d $sp, $sp, -16    # 入栈, 为栈帧分配 16 字节空间

    add.d   $a0, $a0, $a1   # 计算 a0 + a1, 函数返回值存储到 a0 中

    addi.d $sp, $sp, 16     # 出栈, 恢复原来的栈指针
    ld.d $ra, $sp, -8       # 恢复返回地址
    ld.d $fp, $sp, -16      # 恢复栈帧指针
    jr $ra                  # 返回至调用者 main 函数

    .globl  main            # 标记 main 为全局符号
    .type   main, @function # 标记 main 为函数
    .globl  output
    .type   output, @function
output:
    st.d $ra, $sp, -8
    st.d $fp, $sp, -16
    addi.d $fp, $sp, 0
    addi.d $sp, $sp, -16

    # 实际的 output 逻辑，例如打印 $a0 中的值
        。
        。
        。
    #

    addi.d $sp, $sp, 16
    ld.d $ra, $sp, -8
    ld.d $fp, $sp, -16
    jr $ra
main:
    st.d $ra, $sp, -8       # 保存返回地址
    st.d $fp, $sp, -16      # 保存调用者的栈帧指针
    addi.d $fp, $sp, 0      # 设置新的栈帧指针
    addi.d $sp, $sp, -16    # 入栈, 为栈帧分配 16 字节空间

    addi.w  $a0, $zero, 1   # 设置第一个参数
    addi.w  $a1, $zero, 2   # 设置第二个参数
    bl  add                 # 调用 add 函数
    bl  output              # 输出结果
        
    addi.w  $a0, $zero, 0

    addi.d $sp, $sp, 16      # 出栈, 恢复原来的栈指针
    ld.d $ra, $sp, -8        # 恢复返回地址
    ld.d $fp, $sp, -16       # 恢复栈帧指针
    jr $ra                   # 返回至调用者
```

函数中修改了a0，需要注意main返回值，因此可能需要在修改栈帧前修改a0寄存器值为0 ->return 0



##### 五、示例4：学习转值操作和陌生龙芯汇编指令

```c
源程序：
int main (void) {
    int a;
    float b;
    float c;
    a = 8;
    b = a;
    c = 3.5;
    if (b < c)
    {
        return 1;
    }
    return 0;
}


汇编程序如下：
    .text
    .globl  main
    .type   main, @function
main:
    st.d $ra, $sp, -8    # 保存返回地址
    st.d $fp, $sp, -16   # 保存调用者的栈帧指针
    addi.d $fp, $sp, 0   # 设置新的栈帧指针
    addi.d $sp, $sp, -16 # 入栈, 为栈帧分配 16 字节空间

    addi.w  $t0, $zero, 8        # t0 = 8
    movgr2fr.w $ft0, $t0         # 将 0x8 搬运到 $ft0 中
    ffint.s.w $ft0, $ft0         # 将浮点寄存器 $ft0 中存放的定点数转换为浮点格式
    lu12i.w $t1, 0x40600         # $t1[31:0] = 0x40600000, 即浮点数 3.5 的单精度表示
    movgr2fr.w $ft1, $t1         # 将 0x40600000 搬运到 $ft1 中
    fcmp.slt.s $fcc0, $ft0, $ft1 # $fcc0 = ($ft0 < $ft1) ? 1 : 0
    bceqz   $fcc0, .L1           # 如果 $fcc0 等于 0, 则跳转到 .L1 处
    addi.w  $t2, $zero, 1        # $t2 = 1
    b   .L2
.L1:
    addi.w  $t2, $zero, 0 # $t2 = 0
.L2:
    ori  $a0, $t2, 0    # 设置返回值 $a0 = $t2
    addi.d $sp, $sp, 16 # 出栈, 恢复原来的栈指针
    ld.d $ra, $sp, -8   # 恢复返回地址
    ld.d $fp, $sp, -16  # 恢复栈帧指针
    jr $ra              # 返回至调用者
```

这里使用了两种方式来设置浮点寄存器的值：

- 将通用寄存器 `$t0` 中的定点值 `0x8` 通过 `movgr2fr.w` 指令搬运到 `$ft0` 后，通过 `ffint.s.w` 指令转化为浮点值
- 通过 `lu12i.w` 指令将 `$t1` 的 `[31:0]` 位设置为 `3.5` 的单精度表示 `0x40600000`，然后通过 `movgr2fr.w` 指令搬运到 `$ft1`



注意`movgr2fr` 只是换寄存器，还需要紧跟一个转值操作



`fcmp.slt.s $fcc0, $ft0, $ft1`：比较 `$ft0` (8.0f) 是否小于 `$ft1` (3.5f)。因为 `8.0f` 不小于 `3.5f`，所以 `$fcc0` 会被设置为 `0`

`fcmp.cond.s` 的 `cond` 有：

| 助记符 | `cond` |   含义   |         为真的条件         |
| :----: | :----: | :------: | :------------------------: |
| `seq`  | `0x5`  |   相等   |        `$fj == $fk`        |
| `sne`  | `0x11` |   不等   | `$fj > $fk` 或 `$fj < $fk` |
| `slt`  | `0x3`  |   小于   |        `$fj < $fk`         |
| `sle`  | `0x7`  | 小于等于 |        `$fj <= $fk`        |



##### 六、总结龙芯汇编指令和GNU工具链关系以及浮点和整型的互转指令：

###### **龙芯指令（LoongArch）和 GNU 工具链的关系**

**LoongArch** 是中国龙芯中科公司自主研发的一套 CPU 指令集架构（ISA）。它是一个全新的架构，独立于 MIPS、ARM、x86 等现有指令集。

**GNU 工具链** 是一套开源的程序开发工具集合，它包括了：

- **GCC (GNU Compiler Collection)**：编译器，能将 C/C++ 等高级语言代码编译成机器代码。
- **Binutils (GNU Binary Utilities)**：二进制工具集，包括汇编器 (assembler, `as`)、链接器 (linker, `ld`)、归档器 (archiver, `ar`) 等。
- **GDB (GNU Debugger)**：调试器。
- **glibc (GNU C Library)**：标准 C 库。

**关系：**

为了让使用 LoongArch 处理器的计算机能够运行软件，并且让开发者能够为 LoongArch 平台编写和编译程序，**LoongArch 必须得到 GNU 工具链的支持**。

1. **指令集支持**：GNU Binutils 中的汇编器 `as` 必须“懂得”LoongArch 的所有汇编指令（如 `addi.w`, `st.d`, `fcmp.slt.s` 等）。这意味着在 Binutils 中需要添加对 LoongArch 指令集的解析和编码规则。
2. **编译器后端**：GCC 需要一个 LoongArch 的“后端”（backend）。这个后端负责将 GCC 编译器前端生成的中间表示（IR）代码转换成 LoongArch 的机器指令。它要了解 LoongArch 的寄存器约定、函数调用约定（ABI）、指令特性等，以便生成高效、正确的 LoongArch 机器代码。
3. **调试器支持**：GDB 需要了解 LoongArch 的寄存器布局、栈帧结构、指令集等，才能正确地调试运行在 LoongArch 平台上的程序。
4. **C 库支持**：glibc 这样的标准 C 库也需要针对 LoongArch 架构进行编译和优化，提供系统调用接口和运行时支持。



###### **浮点数转整数，整数转浮点的指令**

在 LoongArch 架构中，浮点数和整数之间的转换指令主要有以下几类。这些指令通常遵循 IEEE 754 浮点数标准，并且会涉及到不同的舍入模式。

**整数转浮点数指令 (Integer to Floating-Point)**

这些指令将通用寄存器（或浮点寄存器中被视为整数的值）中的整数转换为浮点数格式，并存储到浮点寄存器中。

- **`ffint.<fmt>.<size>`**: Floating-point Format Integer (signed)
  - **作用**: 将浮点寄存器中存放的**有符号整数**（以 `.w` 或 `.d` 表示其位宽）转换为指定格式（`<fmt>`，如 `.s` 单精度、`.d` 双精度）的浮点数。
  - **例子**:
    - `ffint.s.w $ft0, $ft0`：将 `$ft0` 低 32 位中的 32 位有符号整数转换为单精度浮点数，结果存回 `$ft0`。
    - `ffint.d.d $ft0, $ft0`：将 `$ft0` 中的 64 位有符号整数转换为双精度浮点数，结果存回 `$ft0`。
- **`fufint.<fmt>.<size>`**: Floating-point Unsigned Integer
  - **作用**: 与 `ffint` 类似，但处理的是**无符号整数**。

**注意**: 通常整数会先通过 `movgr2fr.<size>` 指令从通用寄存器移动到浮点寄存器，然后再进行 `ffint` 转换。



 **浮点数转整数指令 (Floating-Point to Integer)**

这些指令将浮点寄存器中的浮点数转换为整数格式，并存储到通用寄存器（或浮点寄存器中被视为整数的值）中。转换过程中涉及到舍入模式。

- **`ftintrm.<size>.<fmt>`**: Floating-point to Integer Round to Minus Infinity (向下取整)
  - **作用**: 将指定格式的浮点数（`<fmt>`）转换为指定位宽的整数（`<size>`），舍入模式为向负无穷大取整（floor）。
  - **例子**: `ftintrm.w.s $ft0, $ft1`：将 `$ft1` 中的单精度浮点数向负无穷大取整后转换为 32 位整数，结果存入 `$ft0` 的低 32 位。
- **`ftintrp.<size>.<fmt>`**: Floating-point to Integer Round to Plus Infinity (向上取整)
  - **作用**: 舍入模式为向正无穷大取整（ceil）。
- **`ftintrz.<size>.<fmt>`**: Floating-point to Integer Round to Zero (向零取整，截断)
  - **作用**: 舍入模式为向零取整（truncate），即简单地丢弃小数部分。
  - **例子**: `ftintrz.w.s $ft0, $ft1`：将 `$ft1` 中的单精度浮点数向零取整后转换为 32 位整数，结果存入 `$ft0` 的低 32 位。
- **`ftintrn.<size>.<fmt>`**: Floating-point to Integer Round to Nearest (四舍五入到最近的整数)
  - **作用**: 舍入模式为四舍五入到最近的整数，如果到两个整数距离相等，通常选择偶数。

**注意**: 转换后的整数值通常会先存储到浮点寄存器中，如果需要将其用于整数运算，还需要通过 `movfr2gr.<size>` 指令从浮点寄存器移动到通用寄存器。



 **寄存器之间数据移动指令 (Register Data Movement)**

这些指令用于在通用寄存器和浮点寄存器之间**按位拷贝数据**，不进行数据格式转换，只是改变数据所在的寄存器类型。

- **`movgr2fr.<size>`**: Move General-purpose Register to Floating-point Register
  - **作用**: 将通用寄存器中的整数值按位拷贝到浮点寄存器。
  - **例子**: `movgr2fr.w $ft0, $t0`：将 `$t0` 的低 32 位拷贝到 `$ft0` 的低 32 位。
- **`movfr2gr.<size>`**: Move Floating-point Register to General-purpose Register
  - **作用**: 将浮点寄存器中的值按位拷贝到通用寄存器。
  - **例子**: `movfr2gr.w $t0, $ft0`：将 `$ft0` 的低 32 位拷贝到 `$t0` 的低 32 位。



### 1.2 了解后端框架

#### 1.2.1 顶层设计

> .
> ├── ...
> ├── include
> │   ├── ...
> │   └── codegen
> │       ├── ASMInstruction.hpp  # 描述汇编指令
> │       ├── CodeGen.hpp         # 后端框架顶层设计
> │       ├── CodeGenUtil.hpp     # 一些辅助函数及宏的定义
> │       └── Register.hpp        # 描述寄存器

顶层的 `Codegen` 类只维护了如下成员：

```c
class CodeGen {
    // ...
  private:
    struct { ... } context;             // 类似 lab2 的 context，用于保存翻译过程中的上下文信息，如当前所在函数
    Module *m;                          // 输入的 IR 模块
    std::list<ASMInstruction> output;   // 生成的汇编指令
};
```

上层函数如下：

在 C++ 中，`explicit` 关键字用于修饰**单参数构造函数**或**除第一个参数外都有默认值的构造函数**。它的主要作用是**禁止隐式类型转换** 

```c
class CodeGen {
    // ...
  public:
    explicit CodeGen(Module *module) : m(module) {} // 构造函数
    std::string print() const;                      // 将汇编指令格式化输出
    void run();                                     // 后端代码生成的入口函数
}
```

需要了解或者实现的是下面一系列函数：

```c
class CodeGen {
    // ...
  private:
    // 栈式分配的变量分配环节，将在函数翻译开始时调用
    void allocate();

    /*=== 以下为助教准备的辅助函数 ===*/
    // 将数据在寄存器和栈帧间搬移。下边的章节将详细介绍
    void load_xxx(...);
    void store_xxx(...);
    // 添加汇编指令
    void append_inst(...);
    // 基本块在汇编程序中的名字
    static std::string label_name(BasicBlock *bb);
    /*=== 以上为助教准备的辅助函数 ===*/

    // 需要补全的部分，进行代码生成的各个环节
    void gen_xxx(...);
};
```

初始代码已经为你处理好了一些繁琐的细节，如全局变量的定义及初始化、汇编中 `section` `type` 的定义等，所以你可以把重点放在栈式分配的实现中



#### 1.2.2 基本类描述

指令类`ASMInstruction`是用来描绘一行汇编指令，在 `CodeGen` 中以 `std::list` 形式组织

`std::list<ASMInstruction> output;`

```c
struct ASMInstruction {
    enum InstType {
        Instruction,    // 汇编指令
        Atrribute,      // 汇编伪指令、描述符等非常规指令
        Label,          // 汇编中的 label
        Comment         // 注释
    } type;             // 用来描述指令的用途，会被下面的 format 函数使用

    std::string content; // 汇编代码，不包含换行符等格式化的信息

    explicit ASMInstruction(std::string s, InstType ty = Instruction); // 构造函数
    std::string format() const; // 根据 type 对 content 进行格式化（如添加缩进、换行符等）
};
```

`ASMInstruction("some debug info", ASMInstruction::Comment)` 定义了一个指令类实例——其用途是注释， `format()` 的返回结果是如下字符串：`"#some debug info\n"`



#### 1.2.3 寄存器类

寄存器分为通用寄存器 `Reg` 、浮点寄存器 `FReg` 和条件标志寄存器 `CFReg`

以下是 `FReg` 的代码定义，`Reg` 与 `CFReg` 的定义与之类似:

```c
struct FReg {
    unsigned id; // 0 <= id <= 31,寄存器的id

    explicit FReg(unsigned i); // 禁止隐式类型转换的构造函数
    bool operator==(const FReg &other);

    std::string print() const;  // 根据 id 返回寄存器别名，如 "$fa0" 而不是 "$f0"

    static FReg fa(unsigned i); // 得到寄存器 $faN
    static FReg ft(unsigned i); // 得到寄存器 $ftN
    static FReg fs(unsigned i); // 得到寄存器 $fsN
};
```

`bool operator==(const FReg &other);`

这是对 `==`（等于）运算符的重载。它允许你直接比较两个 `FReg` 对象是否相等,如 (" `if (r1 = r2) `")

`std::string print() const;`

**`const` 关键字**：表示这个函数是一个“只读”操作，它不会修改 `FReg` 对象自身的 `id` 值

后续还需要具体实现对应的映射关系 `$f0`——>  `$fa0`

- `FReg(0)` 定义了寄存器 `$f0` 的实例，`print()` 的结果是 `"$fa0"`
- 为了获得 `$ft0` 的实例，你可以使用 `FReg(8)`，也可以使用更方便的`FReg::ft(0)`



#### 1.2.4 框架带的辅助函数

`load/store`

**用于方便地提取数据至寄存器和将寄存器数据保存至栈上**

```c
class CodeGen {
    //...
  private:
    // 向寄存器中装载数据
    void load_to_greg(Value *, const Reg &);    // 将 IR 中的 Value 加载到整形寄存器中
    void load_to_freg(Value *, const FReg &);   // 将 IR 中的 Value 加载到浮点寄存器中

    // 将寄存器中的数据保存回栈上
    void store_from_greg(Value *, const Reg &); // 将整形寄存器中的数据保存至 IR 中 Value 对应的栈帧位置
    void store_from_freg(Value *, const FReg &);// 将浮点寄存器中的数据保存至 IR 中 Value 对应的栈帧位置
};
```



对于 12bit 能够表示的整形立即数，直接使用 `$dest = $zero + imm` 的形式（`addi dest, zero, imm`）

对于比较复杂的大立即数提取及浮点立即数提取:

```c
class CodeGen{
    // ...
  private:
    // 向寄存器中加载立即数
    void load_large_int32(int32_t, const Reg &);    // 提取 32 bit 的整数
    void load_large_int64(int64_t, const Reg &);    // 提取 64 bit 的整数
    void load_float_imm(float, const FReg &);       // 提取单精度浮点数（32bit）
};
```



`append_inst()`

`append_inst()` 接口就是用来添加新的 `ASMInstruction`

```c
///按照 ASMInstruction 的构造函数添加指令
append_inst("st.d $ra, $sp, -8", ASMInstruction::Instruction);
// 第二个参数的默认值即为 ASMInstruction::Instruction，所以下边的代码等价
append_inst("st.d $ra, $sp, -8");

///二次封装后的版本
append_inst("st.d", {"$ra", "$sp", "-8"}, ASMInstruction::Instruction);
// 最后一个参数的默认值即为 ASMInstruction::Instruction，所以下边的代码等价
append_inst("st.d", {"$ra", "$sp", "-8"});
```

这些封装的宏定义版本在`include/codegen/CodeGenUtil.hpp`



1.2.5 变量分配

**为程序中的每个变量分配一个栈帧位置，即相对栈指针的一个偏移量**

其设计在`src/codegen/CodeGen.cpp` 中的 `CodeGen::allocate()`

性能上，栈式分配不如寄存器分配，而在实现难度上，栈式分配要简单许多:

​	寄存器分配没有了栈上变量的load到寄存器和结果store回栈上的过程，因此性能更好



**变量分配实现方案：**

- 记录每个变量相对于栈底 `$fp` 的偏移，由于栈从高向低生长，所以这个偏移量为负数
- 固定备份两个寄存器：`$ra` 和 `$fp`
- 备份函数参数： `$fp`把**调用者（父函数）的 `$fp` 值**保存起来，`$ra` 保存返回地址
- 为每个存在定值的指令分配相应的空间
- 对于 `alloca` 指令，`alloca` 本身的定值为指针类型，`alloca` 的空间紧挨着这个这个指针，在更靠近栈顶的位置



```c
// Cminusf源程序：
int main(void) {
    int a;
    a = 1;
    return a;
}

// 生成的 IR 文件（部分）如下：
define i32 @main() {
label_entry:
  %op0 = alloca i32
  store i32 1, i32* %op0
  %op1 = load i32, i32* %op0
  ret i32 %op1
}
```

|         栈帧内容         | 宽度 (byte) | 栈帧位置（相对 `$fp` 的偏移） |
| :----------------------: | :---------: | :---------------------------: |
|          `$ra`           |      8      |              -8               |
|          `$fp`           |      8      |              -16              |
|       `i32* %op0`        |      8      |              -24              |
| `alloca` 出的 `i32` 空间 |      4      |              -28              |
|        `i32 %op1`        |      4      |              -32              |

**得到的汇编指令如下：**

```c
    .text
    .globl main
    .type main, @function
main:
    # prologue
    st.d $ra, $sp, -8
    st.d $fp, $sp, -16
    addi.d $fp, $sp, 0
    addi.d $sp, $sp, -32
.main_label_entry:
    # %op0 = alloca i32
    addi.d $t0, $fp, -28
    st.d $t0, $fp, -24
    # store i32 1, i32* %op0
    ld.d $t0, $fp, -24
    addi.w $t1, $zero, 1
    st.w $t1, $t0, 0
    # %op1 = load i32, i32* %op0
    ld.d $t0, $fp, -24
    ld.w $t0, $t0, 0
    st.w $t0, $fp, -32
    # ret i32 %op1
    ld.w $a0, $fp, -32
    b main_exit
main_exit:
    # epilogue
    addi.d $sp, $sp, 32
    ld.d $ra, $sp, -8
    ld.d $fp, $sp, -16
    jr $ra
```

1/

```c
main:
    # prologue
    st.d $ra, $sp, -8
    st.d $fp, $sp, -16
    addi.d $fp, $sp, 0
    addi.d $sp, $sp, -32

								 +-------------------------+
                 |    调用者栈帧数据       |
                 +-------------------------+  <-- 0x1000  <-- $fp
                 |   备份的 $ra (8字节)    |
                 +-------------------------+  <-- 0x0FF8
                 |  备份的旧 $fp (8字节)   |
                 +-------------------------+  <-- 0x0FF0
                 |                         |
                 |     (为局部变量和       |
                 |      临时值分配的       |
                 |      32字节空间)        |
                 |                         |
                 +-------------------------+  <-- 0x0FE0  <-- $sp (新栈顶)

寄存器状态:
$sp = 0x0FE0  <-- 已更新!
$fp = 0x1000
```

2/	将 `a` 的地址 `0x0FE4` 存入为指针 `%op0` 分配的栈槽 (`$fp - 24`)

```c
    # %op0 = alloca i32
    addi.d $t0, $fp, -28
    st.d $t0, $fp, -24
      
         高地址 ^
                 |
                 +-------------------------+  <-- 0x1000  <-- $fp
                 |    备份的 $ra 和 $fp     |  (共16字节)
                 +-------------------------+  <-- 0x0FF0  ($fp - 16)
                 |   指针 %op0 (8字节)     |
                 | (值为 a 的地址 0x0FE4)  |
                 +-------------------------+  <-- 0x0FE8  ($fp - 24, 指针%op0的地址)
                 |    变量 a 的空间 (4字节)  |
                 +-------------------------+  <-- 0x0FE4  ($fp - 28, 变量a的地址)
                 |  									    |
                 +-------------------------+  <-- 0x0FE0  ($fp - 32) <-- $sp
                 |
          低地址 v
```

3/

```c
    # store i32 1, i32* %op0
    ld.d $t0, $fp, -24
    addi.w $t1, $zero, 1
    st.w $t1, $t0, 0
    
                 +-------------------------+  <-- 0x1000  <-- $fp
                 |       ... (备份)        |
                 +-------------------------+  <-- 0x0FF0
                 | 指针 %op0 (值为0x0FE4)  |
                 +-------------------------+  <-- 0x0FE8
                 |  变量 a 的值 (1)      |  <-- 内存被写入!
      					+-------------------------+  <-- 0x0FE4  ($fp - 28, 变量a的地址)
                 |  									    |
                 +-------------------------+  <-- 0x0FE0  ($fp - 32) <-- $sp
```

4/ 	**指針解引用**`x = *p` 	 `ld.w $t0, $t0, 0`根据 `p` 或 `$t0` 中的地址，去拿回 `a` 的值

```c
    # %op1 = load i32, i32* %op0
    ld.d $t0, $fp, -24
    ld.w $t0, $t0, 0
    st.w $t0, $fp, -32
    # ret i32 %op1
    ld.w $a0, $fp, -32
    b main_exit
    
                 +-------------------------+  <-- 0x1000  <-- $fp
                 |       ... (备份)        |
                 +-------------------------+
                 | 指针 %op0 (值为0x0FE4)  |
                 +-------------------------+
                 |  变量 a 的值 (1)      |
                 +-------------------------+
                 | 临时结果 %op1 (1)     |  <-- 内存被写入!
                 +-------------------------+  <-- 0x0FE0  <-- $sp

寄存器状态:
$a0 = 1  <-- 已准备好返回值!
```

5/

```c
 main_exit:
    # epilogue
    addi.d $sp, $sp, 32
    ld.d $ra, $sp, -8
    ld.d $fp, $sp, -16
    jr $ra
      
      					 +-------------------------+
                 |    调用者栈帧数据       |
                 +-------------------------+  <-- 0x1000  <-- $sp, $fp
                 |   (main的栈帧已释放)    |
                 |   (数据还残留在内存中)  |
                 |          ...            |

寄存器状态:
$sp = 0x1000  <-- 已恢复!
$fp = 0x1000
```

**总结栈帧图：**

```
         高地址 ^
                 |
                 +-------------------------+  <-- 0x1000  <-- $fp
                 |    备份的 $ra 和 $fp     |  (共16字节)
                 +-------------------------+  <-- 0x0FF0  ($fp - 16)
                 |   指针 %op0 (8字节)     |
                 | (值为 a 的地址 0x0FE4)  |
                 +-------------------------+  <-- 0x0FE8  ($fp - 24, 指针%op0的地址)
                 |    变量 a 的空间 (4字节)  |
                 |   (此后将被写入值 1)    |
                 +-------------------------+  <-- 0x0FE4  ($fp - 28, 变量a的地址)
                 |  临时结果 %op1 (4字节)  |
                 |  (此后将被写入值 1)     |
                 +-------------------------+  <-- 0x0FE0  ($fp - 32, 临时结果%op1的地址) <-- $sp
                 |
          低地址 v
```





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