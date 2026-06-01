# 从C语言到ELF二进制格式：关键语法要素的映射



## 编译流程

给出`ed01.c` 和 `ed02.c`

```c
// ed01.c 
extern int val0;
extern int val1;
extern int val2;
extern void func5 ();
extern void func6 ();
extern void func9 (char *);

void func2 () { 
  val0 = 0; 
  val1 = 2; 
  val2 = 7;
}

void func3 () { 
  func5 (); 
  func6 (); 
}

void func7 () { 
  func6 ();
}

void func8 () { 
  func9 ("hello\n");
}
```

```c
// ed02.c 
int val0 = 9;
int val1;
static int val4 = 7;

void func5 () { 
  val4 = 9;
}
```

> clang --target=sparcv9 -c -fPIC -O0 ed01.c
>
> clang --target=sparcv9 -c -fPIC -O0 ed02.c

调用 Clang 编译器驱动，将 C 语言源码编译并汇编成**目标文件（.o 文件）**，并且是在零优化`-O0`的状态下，专门为 SPARC v9 架构生成的。

`-fPIC`:

全称是 **Position-Independent Code（位置无关代码）**。既然我们最终要生成 `libed.so`，这个参数是绝对不可或缺的。它生成的机器码不会写死绝对内存地址，而是使用相对地址。这样，当 `.so` 文件被加载到内存的任何位置时，代码都能正常运行

**产物：**

- `ed01.o`
- `ed02.o`

它们就是包含了 SPARC v9 架构机器码、且带有位置无关特性的**可重定位目标文件（Relocatable Object Files）**



> ld.lld --shared ed01.o ed02.o -o libed.so

在内部做了一次关键的“地址空间重排”，把跨平台编译出来的 SPARC v9 架构的 `ed01.o` 和 `ed02.o` 完美熔合

`.so` 全称是 **Shared Object（共享对象 / 动态链接库）**。在 Linux/Unix 世界里，它本质上就是一个特定格式的 **ELF** 文件（Executable and Linkable Format）。

它与普通的可执行文件（EXE）或静态库（`.a`）相比，有两个特性：

- **共享（Shared）：** 无论你的系统里跑了 10 个还是 100 个需要用到这个库的程序，操作系统在物理内存里只会保留**一份** `libed.so` 的代码段（`.text`）。这 100 个进程通过虚拟内存映射，共同使用这同一份代码，极其变态地节省了物理内存。
- **动态链接（Dynamic Linking）：** 它在编译阶段只是一个半成品。它允许程序在运行前（甚至运行中），才由操作系统的动态链接器（Loader/ld.so）把缺失的函数和变量地址填补上去。



> llvm-readelf -a libed.so > 1.txt

```
Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .dynsym           DYNSYM          0000000000000238 000238 000108 18   A  4   1  8
  [ 2] .gnu.hash         GNU_HASH        0000000000000340 000340 000040 00   A  1   0  8
  [ 3] .hash             HASH            0000000000000380 000380 000060 04   A  1   0  4
  [ 4] .dynstr           STRTAB          00000000000003e0 0003e0 00003a 00   A  0   0  1
  [ 5] .rel.dyn          REL             0000000000000420 000420 000050 10   A  1   0  8
  [ 6] .rel.plt          REL             0000000000000470 000470 000030 10  AI  1  14  8
  [ 7] .rodata           PROGBITS        00000000000004a0 0004a0 000007 01 AMS  0   0  1
  [ 8] .text             PROGBITS        00000000001004a8 0004a8 0000e0 00  AX  0   0  4
  [ 9] .plt              PROGBITS        0000000000200590 000590 0000e0 00 WAX  0   0 16
  [10] .dynamic          DYNAMIC         0000000000300670 000670 0000f0 10  WA  4   0  8
  [11] .got              PROGBITS        0000000000300760 000760 000028 00  WA  0   0  8
  [12] .relro_padding    NOBITS          0000000000300788 000788 001878 00  WA  0   0  1
  [13] .data             PROGBITS        0000000000400788 000788 000008 00  WA  0   0  4
  [14] .got.plt          PROGBITS        0000000000400790 000790 000030 00  WA  0   0  8
  [15] .bss              NOBITS          00000000004007c0 0007c0 000004 00  WA  0   0  4
  [16] .comment          PROGBITS        0000000000000000 0007c0 00003b 01  MS  0   0  1
  [17] .symtab           SYMTAB          0000000000000000 000800 000180 18     19   6  8
  [18] .shstrtab         STRTAB          0000000000000000 000980 00009a 00      0   0  1
  [19] .strtab           STRTAB          0000000000000000 000a1a 00006c 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  R (retain), p (processor specific)
```

`Section Headers:` 这一段。这简直就是 `libed.so` 的楼层导览图。你会清晰地看到：

- **`.text`**：占用最大的空间，里面全都是 `func2`, `func3` 等函数的 SPARC v9 机器码。
- **`.rodata`**：只读数据段，那个孤独的 `"hello\n"` 字符串就藏在这里面。
- **`.data`**：它会有非 0 的大小！因为 `ed02.c` 里的 `int val0 = 9;` 和 `static int val4 = 7;` 实打实地存在这里。
- **`.bss`**：你会发现它的 `Type` 是 `NOBITS`（不占磁盘空间），但它有一个 `Size`。这完美印证了 `int val1;` 这种未初始化变量只占内存不占文件体积的特性。

#### 

```
Symbol table '.dynsym' contains 11 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
     1: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND val2
     2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND func6
     3: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND func9
     4: 00000000001004a8    84 FUNC    GLOBAL DEFAULT     8 func2
     5: 0000000000400788     4 OBJECT  GLOBAL DEFAULT    13 val0
     6: 00000000004007c0     4 OBJECT  GLOBAL DEFAULT    15 val1
     7: 00000000001004fc    28 FUNC    GLOBAL DEFAULT     8 func3
     8: 0000000000100558    48 FUNC    GLOBAL DEFAULT     8 func5
     9: 0000000000100518    20 FUNC    GLOBAL DEFAULT     8 func7
    10: 000000000010052c    44 FUNC    GLOBAL DEFAULT     8 func8

Symbol table '.symtab' contains 16 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS ed01.c
     2: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS ed02.c
     3: 000000000040078c     4 OBJECT  LOCAL  DEFAULT    13 val4
     4: 0000000000300760     0 NOTYPE  LOCAL  HIDDEN     11 _GLOBAL_OFFSET_TABLE_
     5: 0000000000300670     0 NOTYPE  LOCAL  HIDDEN     10 _DYNAMIC
     6: 00000000001004a8    84 FUNC    GLOBAL DEFAULT     8 func2
     7: 0000000000400788     4 OBJECT  GLOBAL DEFAULT    13 val0
     8: 00000000004007c0     4 OBJECT  GLOBAL DEFAULT    15 val1
     9: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND val2
    10: 00000000001004fc    28 FUNC    GLOBAL DEFAULT     8 func3
    11: 0000000000100558    48 FUNC    GLOBAL DEFAULT     8 func5
    12: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND func6
    13: 0000000000100518    20 FUNC    GLOBAL DEFAULT     8 func7
    14: 000000000010052c    44 FUNC    GLOBAL DEFAULT     8 func8
    15: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND func9
```

* **动态链接符号表**和它的超集**全局符号表**

* 找到 `val0`，它的 `Bind` 列会显示 **`GLOBAL`**（全局可见）。

  找到 `val4`，它的 `Bind` 列会显示 **`LOCAL`**（仅局域可见）。 这就是 `static` 关键字的底层真相！编译器通过把符号标记为 `LOCAL`，让其他模块在链接时根本看不见它，从而实现了作用域的隔离。

  找到 `val2` 和 `func6`，你会发现它们的 `Ndx` (Index) 列是 **`UND`** (Undefined)。这说明系统知道它们是外部引用的。

* 一个编译好的so动态链接库文件，其在加载时可能会被分析出依赖其他动态链接库文件，即它其中的符号应用了外部的函数或全局变量。在上述例子中，val2是外部变量，它在ed01.c和ed02.c中都没有定义，只是以extern作了外部符号声明。跨DSO的符号分析，基于.dynsym这个符号表。对比val0和val2在.dynsym section中的字段

* ELF的.dynsym section区分内部和外部符号，用的是Ndx字段。该字段表示的是本符号的实体所在section的编号。上面val0实体位于编号为12的section，即.data section；而val2的Ndx值为UND，意为Undefined，即该符号的实体不在本ELF文件中，为外部符号extern。

  函数func6和func9也是外部符号，没有在ed01.c和ed02.c中定义，只是声明而已。它们的.dynsym section中的Ndx字段同样是UND。

  Ndx字段为UND的这些外部符号，在程序加载时，是需要进行重定位的，即分析出该符号实体的加载地址，然后填入到.got和.got.plt这些全局重定位表section中。关于重定位机制在ELF格式中的具体体现，后面章节会展开讨论。



```
Relocation section '.rel.dyn' at offset 0x420 contains 5 entries:
    Offset             Info             Type               Symbol's Value  Symbol's Name
0000000000300778  0000000000000016 R_SPARC_RELATIVE                  
0000000000300780  0000000000000016 R_SPARC_RELATIVE                  
0000000000300770  0000000100000014 R_SPARC_GLOB_DAT       0000000000000000 val2
0000000000300760  0000000500000014 R_SPARC_GLOB_DAT       0000000000400788 val0
0000000000300768  0000000600000014 R_SPARC_GLOB_DAT       00000000004007c0 val1

Relocation section '.rel.plt' at offset 0x470 contains 3 entries:
    Offset             Info             Type               Symbol's Value  Symbol's Name
00000000004007a8  0000000800000015 R_SPARC_JMP_SLOT       0000000000100558 func5
00000000004007b0  0000000200000015 R_SPARC_JMP_SLOT       0000000000000000 func6
00000000004007b8  0000000300000015 R_SPARC_JMP_SLOT       0000000000000000 func9
```

`Relocation section` 字样的表（通常是 `.rela.plt` 或 `.rela.dyn`）。 Loader处理

这里就是 `libed.so` 留给操作系统的“欠条”。你会在这里看到针对 `val2`, `func6`, `func9` 的重定位条目

**`.rel.dyn`**：数据重定位表。处理全局变量的欠条，加载器会把真实地址填入 GOT（全局偏移表）。

**`.rel.plt`**：函数重定位表。处理函数调用的欠条，加载器会在 PLT（过程链接表）里打补丁，实现正确跳转。





## Q&A

### 1. 如何在文件中找到特定的 Section 或 Program？

**一切尽在 ELF Header。** 操作系统或工具（如 `readelf`）拿到文件后，首先读取文件最开头的 64 个字节（即 ELF Header）。

```
ELF Header:
  Magic:   7f 45 4c 46 02 02 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, big endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file)
  Machine:                           Sparc v9
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          64 (bytes into file)
  Start of section headers:          2696 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         9
  Size of section headers:           64 (bytes)
  Number of section headers:         20
  Section header string table index: 18
There are 20 section headers, starting at offset 0xa88:
```

- **找 Program：** 靠 `Start of program headers: 64 (bytes into file)`。系统直接跳到文件第 64 字节处，就能遍历出 9 个 Program 导览图。

  ```
  Program Headers:
    Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
    PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0001f8 0x0001f8 R   0x8
    LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x0004a7 0x0004a7 R   0x100000
    LOAD           0x0004a8 0x00000000001004a8 0x00000000001004a8 0x0000e0 0x0000e0 R E 0x100000
    LOAD           0x000590 0x0000000000200590 0x0000000000200590 0x0000e0 0x0000e0 RWE 0x100000
    LOAD           0x000670 0x0000000000300670 0x0000000000300670 0x000118 0x001990 RW  0x100000
    LOAD           0x000788 0x0000000000400788 0x0000000000400788 0x000038 0x00003c RW  0x100000
    DYNAMIC        0x000670 0x0000000000300670 0x0000000000300670 0x0000f0 0x0000f0 RW  0x8
    GNU_RELRO      0x000670 0x0000000000300670 0x0000000000300670 0x000118 0x001990 R   0x1
    GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x0
  ```

- **找 Section：** 靠 `Start of section headers: 2696 (bytes into file)`。系统跳到第 2696 字节处，开始解析 20 个 Section 的具体信息（`Off` 列记录了它们在文件中的物理偏移）。

  ```
  Section Headers:
    [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
    [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
    [ 1] .dynsym           DYNSYM          0000000000000238 000238 000108 18   A  4   1  8
    [ 2] .gnu.hash         GNU_HASH        0000000000000340 000340 000040 00   A  1   0  8
    [ 3] .hash             HASH            0000000000000380 000380 000060 04   A  1   0  4
    [ 4] .dynstr           STRTAB          00000000000003e0 0003e0 00003a 00   A  0   0  1
    [ 5] .rel.dyn          REL             0000000000000420 000420 000050 10   A  1   0  8
    [ 6] .rel.plt          REL             0000000000000470 000470 000030 10  AI  1  14  8
    [ 7] .rodata           PROGBITS        00000000000004a0 0004a0 000007 01 AMS  0   0  1
    [ 8] .text             PROGBITS        00000000001004a8 0004a8 0000e0 00  AX  0   0  4
    [ 9] .plt              PROGBITS        0000000000200590 000590 0000e0 00 WAX  0   0 16
    [10] .dynamic          DYNAMIC         0000000000300670 000670 0000f0 10  WA  4   0  8
    [11] .got              PROGBITS        0000000000300760 000760 000028 00  WA  0   0  8
    [12] .relro_padding    NOBITS          0000000000300788 000788 001878 00  WA  0   0  1
    [13] .data             PROGBITS        0000000000400788 000788 000008 00  WA  0   0  4
    [14] .got.plt          PROGBITS        0000000000400790 000790 000030 00  WA  0   0  8
    [15] .bss              NOBITS          00000000004007c0 0007c0 000004 00  WA  0   0  4
    [16] .comment          PROGBITS        0000000000000000 0007c0 00003b 01  MS  0   0  1
    [17] .symtab           SYMTAB          0000000000000000 000800 000180 18     19   6  8
    [18] .shstrtab         STRTAB          0000000000000000 000980 00009a 00      0   0  1
    [19] .strtab           STRTAB          0000000000000000 000a1a 00006c 00      0   0  1
  ```



### 2. 每个 Section 的功能与主要信息是什么？

结合输出中的 `Section Headers`：

- **`.text` (Type: PROGBITS, Flg: AX)**：包含 `func2` 等函数的纯机器指令。`X` 代表可执行。
- **`.rodata` (Type: PROGBITS, Flg: AMS)**：包含 `"hello\n"` 这种只读数据。
- **`.data` (Type: PROGBITS, Flg: WA)**：包含 `val0=9` 和 `val4=7` 这类已初始化的变量。`W` 代表可写。
- **`.bss` (Type: NOBITS, Flg: WA)**：包含未初始化的 `val1`。注意它的 Size 是 4，但在文件中完全不占空间（NOBITS）。
- **`.symtab` / `.dynsym` (Type: SYMTAB/DYNSYM)**：存储函数名、变量名及其内存地址映射的“名片盒”。



### 3. C 源码中的函数和全局变量体现在哪里？

看 `Symbol table '.symtab'`：

```
Symbol table '.symtab' contains 16 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS ed01.c
     2: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS ed02.c
     3: 000000000040078c     4 OBJECT  LOCAL  DEFAULT    13 val4
     4: 0000000000300760     0 NOTYPE  LOCAL  HIDDEN     11 _GLOBAL_OFFSET_TABLE_
     5: 0000000000300670     0 NOTYPE  LOCAL  HIDDEN     10 _DYNAMIC
     6: 00000000001004a8    84 FUNC    GLOBAL DEFAULT     8 func2
     7: 0000000000400788     4 OBJECT  GLOBAL DEFAULT    13 val0
     8: 00000000004007c0     4 OBJECT  GLOBAL DEFAULT    15 val1
     9: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND val2
    10: 00000000001004fc    28 FUNC    GLOBAL DEFAULT     8 func3
    11: 0000000000100558    48 FUNC    GLOBAL DEFAULT     8 func5
    12: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND func6
    13: 0000000000100518    20 FUNC    GLOBAL DEFAULT     8 func7
    14: 000000000010052c    44 FUNC    GLOBAL DEFAULT     8 func8
    15: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND func9
```

- **函数 `func2`**：Value 为 `0x1004a8`，绑定到 `Ndx 8`（正是 `.text` 段）。
- **全局变量 `val0`**：Value 为 `0x400788`，绑定到 `Ndx 13`（正是 `.data` 段）。
- **未初始化变量 `val1`**：绑定到 `Ndx 15`（正是 `.bss` 段）。
- **静态变量 `val4`**：同样绑定到 `Ndx 13`（`.data` 段），但它的 `Bind` 属性是极其特殊的 **`LOCAL`**，这就解释了 `static` 是如何实现作用域隐藏的！



### 4. Section 和 Program Segment之间的关系与数字支撑？

**Section 是链接器视角的原材料，Program 是加载器视角的成品拼盘。** 加载器（Loader）不关心具体的变量和逻辑，只关心“内存权限（读/写/执行）”。因此，它会把具有相同权限的 Section 粗暴地打包成一个 Program Segment。 **数字支撑：** 观察 `Section to Segment mapping`：

**Section（节）**：是给**链接器**看的原材料。它分类极细，比如 `.data` 和 `.bss` 是分开的。

**Program Segment（段）**：是给加载器 (Loader)看的成品拼盘。加载器不关心变量逻辑，只关心“内存权限”。因此它会把具有相同权限的 Section（比如可读可写的 `.data` 和 `.bss`）粗暴地打包成一个 **Segment (RW 权限)** 映射到内存中。

- **Segment 02 (Flg: R E, Read/Execute)**：只打包了 `.text`（代码当然只需读和执行）。
- **Segment 05 (Flg: RW , Read/Write)**：打包了 `.data`, `.got.plt`, `.bss`。它们都是运行时需要动态读写的数据。

```
Program Headers:
  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0001f8 0x0001f8 R   0x8
  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x0004a7 0x0004a7 R   0x100000
  LOAD           0x0004a8 0x00000000001004a8 0x00000000001004a8 0x0000e0 0x0000e0 R E 0x100000
  LOAD           0x000590 0x0000000000200590 0x0000000000200590 0x0000e0 0x0000e0 RWE 0x100000
  LOAD           0x000670 0x0000000000300670 0x0000000000300670 0x000118 0x001990 RW  0x100000
  LOAD           0x000788 0x0000000000400788 0x0000000000400788 0x000038 0x00003c RW  0x100000
  DYNAMIC        0x000670 0x0000000000300670 0x0000000000300670 0x0000f0 0x0000f0 RW  0x8
  GNU_RELRO      0x000670 0x0000000000300670 0x0000000000300670 0x000118 0x001990 R   0x1
  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x0
```

```
Section to Segment mapping:
  Segment Sections...
   00     
   01     .dynsym .gnu.hash .hash .dynstr .rel.dyn .rel.plt .rodata 
   02     .text 
   03     .plt 
   04     .dynamic .got .relro_padding 
   05     .data .got.plt .bss 
   06     .dynamic 
   07     .dynamic .got .relro_padding 
   08     
   None   .comment .symtab .shstrtab .strtab 
```



### 5. ELF Header 中的枚举型还有哪些可能的值？

- **Class:** `ELF64` (64位) 或 `ELF32` (32位)。
- **Data:** `2's complement, big endian` (大端序，如 SPARC/PowerPC) 或 `little endian` (小端序，如 x86/ARM)。
- **Type:** `DYN` (动态库 .so)，`EXEC` (可执行文件 .exe)，`REL` (可重定位目标文件 .o)，`CORE` (崩溃核心转储)。
- **Machine:** `Sparc v9`，`x86_64` (AMD64)，`AArch64` (ARM64) 等。

```
ELF Header:
  Magic:   7f 45 4c 46 02 02 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, big endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file)
  Machine:                           Sparc v9
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          64 (bytes into file)
  Start of section headers:          2696 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         9
  Size of section headers:           64 (bytes)
  Number of section headers:         20
  Section header string table index: 18
There are 20 section headers, starting at offset 0xa88:
```



### 6. extern 标记的外部变量与本地变量的区别？

在 `.symtab` 或 `.dynsym` 中极其明显：

- **本地变量（如 `val0`）**：有确切的 `Value` (0x400788)，有 `Size` (4)，且归属一个真实的房间号 `Ndx` (13)。
- **extern 变量（如 `val2`, `func6`）**：`Value` 为 0，`Size` 为 0，最关键的是它的房间号 **`Ndx` 是 `UND` (Undefined)**。这表示：“我认识这个符号，但它不在我家，得去外面找”。



### 7. ELF 信息中数字的含义体系？

- **Offset / Off**：**文件相对偏移**。纯物理概念，表示在 `.so` 磁盘文件里的第几个字节。
- **Address / VirtAddr**：**虚拟地址**。这是操作系统的内存幻觉，表示这个段被加载到内存后，系统希望它位于哪个虚拟地址空间。
- **Size / FileSiz / MemSiz**：大小。注意，对于 `.bss`，`FileSiz` 通常为 0，但 `MemSiz` 会是一个正数。
- **Ndx (Index)**：**序号索引**。相当于表里的行号，用于快速关联其他的表项。



### 8. 为什么有些 Section 没有展开描述？

因为 `llvm-readelf` 只是一个**结构解析器**，它负责展示 ELF 文件的骨架和表单元数据。对于 `.text`（机器码）、`.rodata`（字符串明文）这类属于 `PROGBITS`（程序核心数据）的内容，它无法直接用表格呈现。 要看里面的血肉，必须换工具：用 `llvm-objdump -d` 进行汇编反编译看代码，或者用 `llvm-objdump -s` 打印 Hex 原始十六进制数据看变量



### 9. rel.dyn 和 rel.plt 的作用与 Type 位域？

它们是动态链接库留给操作系统的“欠条合集”。

- **`.rel.dyn`（数据重定位）**：处理全局变量的欠条。输出中针对 `val0`, `val1`, `val2` 的 `Type` 是 `R_SPARC_GLOB_DAT`。加载器看到这个标记，就会把它们在内存里的真实地址，填入 GOT（全局偏移表）中。
- **`.rel.plt`（函数重定位）**：处理函数调用的欠条。输出中针对 `func5`, `func6`, `func9` 的 `Type` 是 `R_SPARC_JMP_SLOT`。加载器看到这个，就会在 PLT（过程链接表）里打补丁，让程序调用外部函数时能跳转到正确的内存空间。



### 10. global和static实体的差异

global和static符号在ELF文件中的差异体现在Bind字段上，global变量会被标记为GLOBAL，而static符号会被标记为LOCAL，意为本地符号，因此标记了static的全局变量或函数，只能被本源文件中的函数或变量访问，而不能跨C源文件访问。





## 其他

### 汇编反编译

> llvm-objdump -S libed.so > 2.asm

可以发现ELF文件的.text section是C源码翻译成的汇编指令流



### 函数局部变量

函数的局部变量不同于全局变量，它们不会直接体现在ELF文件的某个section中。局部变量的实体存在，仅在程序运行阶段，它对应于各个函数中分配出的**栈空间的一个slot**。在某个函数开始运行，栈空间得以分配，相应的局部变量的生命周期开始；该函数运行结束，相应的栈空间被回收，该局部变量的生命周期就完全结束了。当然，也可以更精确的说，局部变量的生命周期从它被声明开始，到它不再被任何语句访问结束。



### 绝对寻址和PC相对寻址

**绝对寻址 (Absolute Addressing)：简单粗暴的“刻舟求剑”**

- **本质：** 直接把目标变量或函数的内存绝对地址（比如 `0x4000`）硬编码写死在指令里。具体实现可能由一个寄存器、两个寄存器联合，或者寄存器加立即数来体现。
- **致命弱点：** 它**完全不满足 PIC（位置无关代码）的要求**。一旦这个程序被操作系统加载到内存的另一个基地址，原本写死的 `0x4000` 就成了错的。
- **归宿：** 只能用于老老实实的**静态链接场景**，因为静态链接在运行前地址就绝对固定了。

**PC 相对寻址 (PC-Relative Addressing)：灵活的“相对论”**

- **本质：** 不写死目标的绝对地址，而是记录目标和当前指令（PC 寄存器）之间的**距离（Offset/偏移量）**。
- **局限中的灵活：** 只要调用者（指令）和被调用者（目标符号）被打包在同一个文件（EXE 或 DSO）内部，它们之间的相对距离就是永远不变的。因此，它能够满足这种内部相互访问时的 PIC 要求。

**绝对寻址**：硬编码目标的绝对物理地址。优点是省去查表的性能损耗，极其快；致命缺点是**不支持位置无关代码（PIC）**，只能用于静态链接。

**PC 相对寻址**：记录目标符号与当前指令（PC）的相对偏移距离。只要调用者和被调用者在同一个文件内，距离就是固定的，因此**能够满足同一文件内部访问的 PIC 要求**。

*(注：如果使用 `-fPIC` 编译动态库，为了实现跨模块的位置无关，编译器会放弃这两种简单的寻址，被迫使用更复杂的 **GOT/PLT 查表机制**。)*



举例`nonpic_reldemo.c`

```c
int val0;
void func2() {
    val0 = 9;
}
void func3() {
    func2();
}
```

> clang --target=sparcv9 -c -O0 nonpic_reldemo.c

此处不能采用PIC方式（使用参数-fPIC）编译，否则LLVM的Sparc后端会放弃绝对寻址和PC相对寻址，转而使用全局偏移表和函数跳转表的重定位方式

> ld.lld nonpic_reldemo.o -o nonpic_reldemo -e func3
>
> llvm-readelf -a nonpic_reldemo > 3.txt

```
Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .text             PROGBITS        0000000000200158 000158 000034 00  AX  0   0  4
  [ 2] .bss              NOBITS          000000000030018c 00018c 000004 00  WA  0   0  4
  [ 3] .comment          PROGBITS        0000000000000000 00018c 00003b 01  MS  0   0  1
  [ 4] .symtab           SYMTAB          0000000000000000 0001c8 000078 18      6   2  8
  [ 5] .shstrtab         STRTAB          0000000000000000 000240 00002f 00      0   0  1
  [ 6] .strtab           STRTAB          0000000000000000 00026f 000023 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  R (retain), p (processor specific)
```

```
Symbol table '.symtab' contains 5 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS nonpic_reldemo.c
     2: 0000000000200158    32 FUNC    GLOBAL DEFAULT     1 func2
     3: 000000000030018c     4 OBJECT  GLOBAL DEFAULT     2 val0
     4: 0000000000200178    20 FUNC    GLOBAL DEFAULT     1 func3
There are no section groups in this file.
```

> llvm-objdump -S nonpic_reldemo > 4.asm

```assembly

nonpic_reldemo:	file format elf64-sparc

Disassembly of section .text:

0000000000200158 <func2>:
  200158: 9d e3 bf 80  	save %sp, -0x80, %sp
  20015c: 31 00 00 00  	sethi 0x0, %i0
  200160: b0 06 23 00  	add %i0, 0x300, %i0
  200164: b1 2e 30 0c  	sllx %i0, 0xc, %i0
  200168: b2 10 20 09  	mov	0x9, %i1
  20016c: f2 26 21 8c  	st %i1, [%i0+0x18c]
  200170: 81 c7 e0 08  	ret
  200174: 81 e8 00 00  	restore

0000000000200178 <func3>:
  200178: 9d e3 bf 50  	save %sp, -0xb0, %sp
  20017c: 7f ff ff f7  	call 0x200158
  200180: 01 00 00 00  	nop
  200184: 81 c7 e0 08  	ret
  200188: 81 e8 00 00  	restore

```

从反汇编和ELF信息中func2、func3和val0几个符号的地址可以看出:

- func2对val0的访问使用了绝对寻址重定位。从反汇编看，这一段用连续三条sethi/add/sllx将寄存器%i0赋值为为0x300000，然后在执行store（助记符st）时，用%i0加上立即数396构造出了0x30018C这个地址。参考上面的ELF信息，0x30018C正是val0这个全局变量的地址。
- func3对func2的调用，使用的是PC相对寻址重定位方式。注意func3中call -36这条指令，其地址为0x20017c，减去十进制的36，结果正是func2的入口0x200158。

