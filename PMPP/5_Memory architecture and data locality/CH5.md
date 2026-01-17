# CH5 Memory Architecture and Data Locality

## 5.1 Importance of memory access efficiency (内存访问效率的重要性)

**Q1: 什么是“计算与全局内存访问比” (Compute to Global Memory Access Ratio, CGMA)?**

- **A:** 它定义为程序中每从全局内存读取 1 个字节 (Byte) 所执行的浮点运算次数 (FLOPs)。也称为**计算强度 (Arithmetic Intensity)**。
- **公式:** $\text{CGMA} = \frac{\text{Floating Point Operations}}{\text{Bytes Accessed from Global Memory}}$

**Q2: 为什么基础的矩阵乘法 Kernel (无 Tiling) 效率很低？**

- **A:** 因为它的 CGMA 比率极低。基础算法中，每进行 2 次浮点运算（1乘1加），需要从全局内存读 8 字节（2个 float）。比率为 $2/8 = 0.25 \text{ FLOP/B}$。
- **后果:** 对于像 A100 这样带宽 1555 GB/s 的 GPU，其最大吞吐量被限制在 $1555 \times 0.25 = 389 \text{ GFLOPS}$，仅为硬件峰值性能的 **2%** 左右。这种程序被称为 **Memory-bound (受限于内存)** 程序。

**Q3: 什么是 Roofline Model (屋顶模型)?**

- **A:** 它是一个直观的性能模型。
  - **斜线部分:** 代表 Memory-bound 区域，性能受限于内存带宽。
  - **水平线部分:** 代表 Compute-bound 区域，性能受限于计算核心峰值。
  - **目标:** 通过提高计算强度（向右移动），让程序从斜线区进入水平线区。

------

## 5.2 CUDA memory types (CUDA 内存类型)

**Q1: 请简述 CUDA 的主要内存类型及其物理位置和速度。**

- **A:** 参见下表:

| **内存类型** | **关键字**     | **物理位置**         | **速度**        | **作用域** | **生命周期**  |
| ------------ | -------------- | -------------------- | --------------- | ---------- | ------------- |
| **Register** | (自动标量)     | 片上 (On-chip)       | 最快            | Thread     | Thread        |
| **Shared**   | `__shared__`   | 片上 (On-chip)       | 极快            | Block      | Kernel (Grid) |
| **Global**   | `__device__`   | 片下 (Off-chip DRAM) | 慢              | Grid       | Application   |
| **Local**    | (自动数组)     | 片下 (Off-chip DRAM) | 慢              | Thread     | Thread        |
| **Constant** | `__constant__` | 片下 (有片上缓存)    | 快 (缓存命中时) | Grid       | Application   |

**Q2: 为什么访问寄存器比访问全局内存快得多？**

- **A:**
  1. **物理距离:** 寄存器就在处理器核心内部，而全局内存位于芯片外部。
  2. **指令开销:** 算术指令直接使用寄存器作为操作数（如 `fadd r1, r2, r3`），不需要额外的 Load/Store 指令。而访问全局内存需要专门的 Load 指令去搬运数据。
  3. **带宽与能耗:** 寄存器带宽极高且能耗极低。

**Q3: 什么是本地内存 (Local Memory)？它快吗？**

- **A:** 不要被名字骗了。它是用来存放线程私有的大型数组或寄存器溢出数据的。虽然作用域是局部的，但它**物理上存储在 Global Memory 中**，所以它很慢。

------

## 5.3 Tiling for reduced memory traffic (利用分块减少内存流量)

**Q1: 什么是 Tiling (分块)？它的核心原理是什么？**

- **A:** Tiling 是一种将数据划分为小块 (Tile) 的策略，使得每个 Tile 都能放入高速的 **Shared Memory** 中。
- **原理:** 利用数据局部性 (Locality)。线程协作将一个 Tile 的数据从 Global Memory 搬运到 Shared Memory，然后所有线程在 Shared Memory 上多次复用这些数据，从而减少去 Global Memory 搬运的次数。

**Q2: 对于矩阵乘法，使用边长为 $N$ 的 Tile 能带来多大的性能提升？**

- **A:** 全局内存的访问流量将减少 **$N$ 倍**。
  - 例如：使用 $16 \times 16$ 的 Tile，Global Memory 访问量减少为原来的 **1/16**。计算强度从 0.25 FLOP/B 提升到 4.0 FLOP/B。

**Q3: Tiling 算法的执行流程通常是怎样的？**

- **A:** 分阶段 (Phase) 执行。
  - **Phase 0:** 协作加载第一个 Tile -> `__syncthreads()` -> 计算点积的一部分 -> `__syncthreads()`。
  - **Phase 1:** 协作加载下一个 Tile -> ...
  - 直到遍历完所有 Tile。

------

## 5.4 A tiled matrix multiplication kernel (分块矩阵乘法 Kernel)

**Q1: 在 Tiled Kernel 中，为什么需要两个 `__syncthreads()`？**

- **A:**
  1. **第一个 (Load 之后):** 解决 **Read-After-Write (RAW)** 依赖。确保所有线程都把自己负责的数据搬进了 Shared Memory，大家才能开始计算。
  2. **第二个 (Compute 之后):** 解决 **Write-After-Read (WAR)** 依赖。确保所有线程都算完了当前 Tile 的数据，才能进入下一轮加载新数据，防止旧数据被过早覆盖。

**Q2: 如何计算协作加载时的 Global Memory 索引？**

- **A:** 假设当前是第 `ph` 个阶段：

  - 加载 M (行矩阵): 行不变 (Row)，列随阶段移动 (ph*TILE_WIDTH + tx)。

    Index_M = Row * Width + (ph * TILE_WIDTH + tx)。

  - 加载 N (列矩阵): 列不变 (Col)，行随阶段移动 (ph*TILE_WIDTH + ty)。

    Index_N = (ph * TILE_WIDTH + ty) * Width + Col。

------

## 5.5 Boundary checks (边界检查)

**Q1: 当矩阵尺寸不是 Tile 大小的整数倍时，会发生什么问题？**

- **A:**
  1. **加载越界:** 线程会尝试读取矩阵外部的内存地址，导致读取无效数据或程序崩溃 (Segfault)。
  2. **计算错误:** 如果不处理，读进来的垃圾值会污染最终的点积结果。

**Q2: 如何处理边界问题？**

- **A:** 使用 **Padding with 0 (补零策略)**。
  - 在加载数据到 Shared Memory 时，检查当前索引是否在矩阵范围内：
    - `if (Row < Width && Col < Width)`: 正常加载。
    - `else`: 往 Shared Memory 填入 **0.0**。
  - 这样，即使进行乘加运算，0.0 也不会影响最终结果。

**Q3: 写回结果 P 时需要检查吗？**

- **A:** 需要。只有当线程负责的坐标 `(Row, Col)` 在矩阵范围内时，才执行写入操作，否则该线程不应修改 Global Memory。

------

## 5.6 Impact of memory usage on occupancy (内存使用对占用率的影响)

**Q1: 为什么 Shared Memory 用多了会降低性能？**

- **A:** 每个 SM 的 Shared Memory 总容量是有限的（如 A100 为 164KB）。如果每个 Block 占用的 Shared Memory 太多，SM 能同时运行的 Block 数量就会减少，导致 **Occupancy (占用率)** 下降，从而降低 GPU 掩盖延迟的能力。

**Q2: 什么是动态共享内存 (Dynamic Shared Memory)？如何使用？**

- **A:** 允许在运行时（而不是编译时）决定 Shared Memory 的大小。
  1. **Kernel 声明:** `extern __shared__ float s_mem[];` (不指定大小)。
  2. **Host 启动:** `kernel<<<grid, block, shared_mem_size>>>(...)` (在第3个参数传入字节数)。
  3. **用途:** 使得 Kernel 可以根据硬件能力或问题规模自适应调整 Tile 大小，不需要重新编译。

------

## 5.7 Summary (总结)

**Q1: 本章的核心思想是什么？**

- **A:**
  1. **局部性 (Locality):** 高性能并行计算必须利用数据局部性。
  2. **Tiling:** 是在 GPU 上显式利用 Shared Memory 实现局部性的主要手段。
  3. **权衡:** 程序员必须在利用快速内存（Shared Mem/Regs）和避免资源超限（保持高 Occupancy）之间找到平衡。

------

## 5.8 Exercise

### **Exercise 1: 矩阵加法 (Matrix Addition) 与共享内存**

Q: 考虑矩阵加法。能否利用共享内存 (Shared Memory) 来减少全局内存 (Global Memory) 的带宽消耗？

A: 不能 (No)。

**解析:**

1. **数据复用 (Data Reuse) 是关键**: 共享内存之所以能减少全局内存带宽，是因为它允许我们将数据从全局内存搬运一次，然后在片上（On-chip）被多个线程多次复用。
2. **矩阵加法特性**: 矩阵加法计算公式为 $C[i][j] = A[i][j] + B[i][j]$。
   - 每个输入元素 $A[i][j]$ 和 $B[i][j]$ 仅被一个线程读取**一次**。
   - 计算结果 $C[i][j]$ 仅被写入**一次**。
3. **结论**: 由于没有任何输入数据被其他线程重复使用，使用 Tiling（分块）将其搬入共享内存只会增加代码复杂度，而不会减少全局内存的访问次数。每个数据还是必须从 Global Memory 读进来一次。

------

### **Exercise 2: 8x8 矩阵乘法的 Tiling 图解**

**Q:** 画出 8x8 矩阵乘法在使用 2x2 Tiling 和 4x4 Tiling 时的等效图（类似 Fig 5.7），并验证带宽减少量与 Tile 大小成正比。

**A:**

- **无分块 (No Tiling):** 每个线程读取一行 ($N=8$) 和一列 ($N=8$)。总读取量 $\propto 8+8 = 16$。
- **2x2 Tiling:**
  - 将矩阵切分为 $2 \times 2$ 的块。
  - 计算需要 $8/2 = 4$ 个 Phase。
  - 每个 Phase，Block 读取 $2 \times 2$ 个 M 元素和 $2 \times 2$ 个 N 元素。
  - 每个线程在每个 Phase 平均读取 $1$ 个 M 和 $1$ 个 N。总共读取 $4 \text{ (Phases)} \times 2 = 8$ 次。
  - **减少倍数:** $16 / 8 = \mathbf{2}$ (等于 Tile Width)。
- **4x4 Tiling:**
  - 将矩阵切分为 $4 \times 4$ 的块。
  - 计算需要 $8/4 = 2$ 个 Phase。
  - 每个 Phase，每个线程读取 $1$ 个 M 和 $1$ 个 N。总共读取 $2 \text{ (Phases)} \times 2 = 4$ 次。
  - **减少倍数:** $16 / 4 = \mathbf{4}$ (等于 Tile Width)。

**结论:** 全局内存带宽的减少倍数确实等于 **Tile 边长 (TILE_WIDTH)**。

------

### **Exercise 3: 忘记 `__syncthreads()` 的后果**

**Q:** 如果在 Fig 5.9 的 Kernel 中忘记写某一个或两个 `__syncthreads()`，会发生什么错误？

**A:**

1. **缺少第一个 `__syncthreads()` (Line 21, Load 之后):**
   - **错误类型:** **Read-After-Write (RAW) 数据竞争**。
   - **后果:** 某些跑得快的线程可能在其他线程完成数据搬运（Load）之前就开始计算。它们会读取到共享内存中未初始化或上一轮残留的错误数据。
2. **缺少第二个 `__syncthreads()` (Line 26, Compute 之后):**
   - **错误类型:** **Write-After-Read (WAR) 数据竞争**。
   - **后果:** 某些跑得快的线程可能已经完成了当前 Phase 的计算，进入下一个 Phase 并开始加载新数据，覆盖了共享内存中的旧数据。而跑得慢的线程可能还在使用旧数据进行计算，导致结果出错。

------

### **Exercise 4: 共享内存 vs 寄存器**

**Q:** 假设容量不是问题，为什么用共享内存 (Shared Memory) 存 Global Memory 取来的数据比用寄存器 (Registers) 更有价值？

A:

核心原因是：数据共享 (Data Sharing) 与 协作 (Collaboration)。

- **寄存器是私有的:** 寄存器中的数据只有拥有它的那个线程能看得到。如果你把数据加载到寄存器，其他线程无法访问它。
- **共享内存是可见的:** 共享内存对同一 Block 内的所有线程可见。
- **价值:** 在 Tiling 算法中，我们希望 Thread A 从 Global Memory 读进来的数据（比如 $M[0][1]$），能被 Thread B、Thread C 等复用。只有将数据放入共享内存，才能实现这种**“一人干活，全家受益”**的协作模式，从而减少重复去 Global Memory 搬运的次数。

------

### **Exercise 5: 32x32 Tiling 的带宽减少**

Q: 使用 32x32 的 Tile，输入矩阵 M 和 N 的带宽使用减少了多少？

A: 减少了 32 倍 (factor of 32)。

(即现在的带宽消耗是原来的 1/32。)

------

### **Exercise 6: 局部变量 (Local Variable) 的版本数量**

Q: 1000 个 Block，每 Block 512 个线程。一个局部变量 (Local Variable) 会有多少个版本？

A: 512,000 个。

解析: 局部变量（自动变量）存储在寄存器或 Local Memory 中，其作用域是单个线程 (Per Thread)。



$$1000 \text{ Blocks} \times 512 \text{ Threads/Block} = 512,000 \text{ Threads}$$



每个线程都有自己私有的一份副本。

------

### **Exercise 7: 共享变量 (Shared Memory Variable) 的版本数量**

Q: 接上题，如果变量声明为 __shared__，会有多少个版本？

A: 1000 个。

解析: 共享内存变量的作用域是线程块 (Per Block)。每个 Block 共享同一份副本。



$$1000 \text{ Blocks} = 1000 \text{ Versions}$$

------

### **Exercise 8: 矩阵乘法 Global Memory 请求次数**

Q: $N \times N$ 矩阵乘法，每个元素从 Global Memory 被请求多少次？

A:

- **a. 无 Tiling:** 每个结果元素需要读取一行 M ($N$个) 和一列 N ($N$个)。
  - 总请求次数 = $N^2 \text{ (结果元素)} \times 2N \text{ (读M+N)} = \mathbf{2N^3}$。
  - (每个输入元素平均被读 $N$ 次)。
- **b. Tiling ($T \times T$):** 带宽减少了 $T$ 倍。
  - 总请求次数 = $\mathbf{\frac{2N^3}{T}}$。
  - (每个输入元素平均被读 $N/T$ 次)。

------

### **Exercise 9: Compute-bound vs Memory-bound 判断**

Q: Kernel 属性：每线程 36 FLOPs，7 个 32-bit (4 Byte) Global Memory 访问。

计算强度 (Arithmetic Intensity):



$$\frac{36 \text{ FLOPs}}{7 \times 4 \text{ Bytes}} = \frac{36}{28} \approx \mathbf{1.29 \text{ FLOP/B}}$$

**A:**

- **a. Peak 200 GFLOPS, BW 100 GB/s:**
  - 硬件拐点 (Ridge Point) = $200 / 100 = 2.0 \text{ FLOP/B}$。
  - $1.29 < 2.0$，落在斜坡区。
  - **结论: Memory-bound**。
- **b. Peak 300 GFLOPS, BW 250 GB/s:**
  - 硬件拐点 (Ridge Point) = $300 / 250 = 1.2 \text{ FLOP/B}$。
  - $1.29 > 1.2$，落在平台区。
  - **结论: Compute-bound**。

------

### **Exercise 10: 矩阵转置 Kernel 分析**

Q: 分析转置 Kernel (Fig 5.15) 的正确性。

A:

1. **错误原因:** 代码中存在 **Read-After-Write (RAW) 竞争**，且没有 `__syncthreads()`。
   - `blockA[threadIdx.y][threadIdx.x] = ...` (写入)
   - `... = blockA[threadIdx.x][threadIdx.y]` (读取)
   - 如果是转置操作，Thread(0,1) 会写入 `blockA[1][0]`，然后读取 `blockA[0][1]`。
   - Thread(1,0) 会写入 `blockA[0][1]`，然后读取 `blockA[1][0]`。
   - **竞争:** Thread(0,1) 读取 `blockA[0][1]` 时，必须确保 Thread(1,0) 已经写进去了。如果没有同步，顺序是不确定的。
2. **a. 哪些 BLOCK_SIZE 能正确运行?**
   - 如果 `BLOCK_SIZE` 的总线程数 $\le$ Warp Size (32)，并且架构保证 Warp 内同步执行（Lock-step），可能侥幸正确（但不推荐依赖）。
   - 对于一般情况，**没有任何 Block Size 能保证绝对正确**，除非加同步。
3. **b. 修复方法:**
   - 在写入 Shared Memory 之后，读取之前，插入 **`__syncthreads();`**。

------

### **Exercise 11: `foo_kernel` 代码分析**

Q: 分析代码中的变量版本和资源。

A:

- **a. `i` 的版本数:** 自动变量 (Register)，Per Thread。Grid Dim = 8 Blocks, Block Dim = 128. Total Threads = 1024. -> **1024 个**。
- **b. `x[]` 的版本数:** 自动数组 (Local Mem/Reg)，Per Thread。-> **1024 个**。
- **c. `y_s` 的版本数:** `__shared__`，Per Block。-> **8 个**。
- **d. `b_s[]` 的版本数:** `__shared__`，Per Block。-> **8 个**。
- **e. 每 Block 共享内存大小:**
  - `y_s` (float, 4B) + `b_s[128]` (128 * 4B = 512B) = **516 Bytes**。
- **f. FLOP/Global Memory Access Ratio:**
  - **Global Access (Bytes):**
    - `x[j] = a[...]`: 循环 4 次，读 4 个 float = 16B。
    - `b_s[...] = b[i]`: 读 1 个 float = 4B。
    - `b[i] = ...`: 写 1 个 float = 4B。
    - 总计: $16 + 4 + 4 = 24$ Bytes。
  - **Floating Point Ops:**
    - `2.5f*x[0] + ...`: 4 次乘法 + 3 次加法。
    - `+ y_s * b_s[...]`: 1 次乘法 + 1 次加法。
    - `+ b_s[...]`: 1 次加法。
    - 总计: 5 Mul + 5 Add = 10 FLOPs。
  - **Ratio:** $10 \text{ FLOPs} / 24 \text{ Bytes} \approx \mathbf{0.42 \text{ FLOP/B}}$。

------

### **Exercise 12: 占用率 (Occupancy) 计算**

**硬件限制:** 2048 threads/SM, 32 blocks/SM, 64K Regs/SM, 96KB Smem/SM。

**a. Kernel: 64 threads/block, 27 regs, 4KB smem**

- **按寄存器限制:** $65536 / (64 \times 27) = 37.9 \to 37$ Blocks。 (受限于 32 Blocks/SM 硬件上限 $\to$ **32 Blocks**)
- **按共享内存限制:** $96 \text{KB} / 4 \text{KB} = \mathbf{24 \text{ Blocks}}$。
- **按线程限制:** $2048 / 64 = 32$ Blocks。
- **最终活跃 Blocks:** **24** (被 Shared Memory 限制)。
- **活跃线程数:** $24 \times 64 = 1536$。
- **Full Occupancy?** No ($1536 < 2048$)。
- **Limiting Factor:** **Shared Memory**。

**b. Kernel: 256 threads/block, 31 regs, 8KB smem**

- **按寄存器限制:** $65536 / (256 \times 31) = 8.25 \to \mathbf{8 \text{ Blocks}}$。
- **按共享内存限制:** $96 \text{KB} / 8 \text{KB} = 12$ Blocks。
- **按线程限制:** $2048 / 256 = \mathbf{8 \text{ Blocks}}$。
- **最终活跃 Blocks:** **8**。
- **活跃线程数:** $8 \times 256 = 2048$。
- **Full Occupancy?** **Yes** (2048 threads, 100%)。