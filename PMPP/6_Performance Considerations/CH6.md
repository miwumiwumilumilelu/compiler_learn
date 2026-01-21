# CH6 Performance Considerations



### 6.1 Memory Coalescing (内存合并)

**Q1: 为什么 DRAM 的物理特性决定了我们需要做“内存合并”？**

- **A:** DRAM 的读取速度很慢（几十纳秒），但它支持 **Burst (突发传输)** 模式。当读取一个地址时，DRAM 实际上会激活并传输该地址周围的一连串数据。
- **复习点:** 如果线程只用了 Burst 中的一个字节而丢弃其余的，带宽利用率会极低；合并访问能充分利用整个 Burst。

**Q2: 什么是 Coalesced Access (合并访问) 的黄金标准？**

- **A:** 当一个 Warp 中的 32 个线程执行加载指令时，它们访问的内存地址是**连续的**（consecutive global memory locations）。
  - 例如：Thread $k$ 访问地址 $X + k$。
  - 此时硬件会将这 32 个请求合并成一个或极少数几个内存事务。

**Q3: 在处理矩阵时，Row-Major 和 Column-Major 对 Coalescing 有什么影响？**

- **A:** C 语言使用 **Row-Major (行优先)**。
  - **好模式:** 线程索引 `tx` 映射到列 (`col`)，即访问同一行的相邻元素 (`M[row][tx]`)。这在物理上是连续的。
  - **坏模式:** 线程索引 `tx` 映射到行 (`row`)，即跨行访问 (`M[tx][col]`)。虽然逻辑上只差一行，但物理地址相差了 `Width`，导致 **Strided Access (跨步访问)**，无法合并。

**Q4: 如果算法逻辑必须按列读取（例如矩阵 B 是列优先存储），如何实现合并？请解释 "Corner Turning"。**

- **A:** 使用 **Shared Memory** 作为中转站，这就是 **Corner Turning (转角优化)**。
  1. **加载阶段:** 改变线程的分工。让线程按照**物理地址连续**的方式（即沿着列的方向）去读取 Global Memory，确保存取是合并的。
  2. **存入阶段:** 将数据存入 Shared Memory。
  3. **计算阶段:** 线程从 Shared Memory 中按需读取（此时乱序读也没关系，因为 Shared Memory 是 SRAM，不需要 Coalescing）。

------

### 6.2 Hiding Memory Latency (掩盖内存延迟)

**Q1: 既然 DRAM 这么慢，现代 GPU 是如何提升带宽利用率的？（硬件层面）**

- **A:** 使用 **Channels (通道)** 和 **Banks (存储体)** 的并行结构。
  - **Channels:** 相当于多条高速公路。
  - **Banks:** 每个通道连接多个 Banks。当一个 Bank 在忙着解码/充电（Latency）时，另一个 Bank 可以利用总线传输数据。

**Q2: 什么是 "Interleaved Data Distribution" (交错数据分布)？**

- **A:** 硬件会自动将连续的数组元素打散存储到不同的 Channels 和 Banks 中（例如 `M[0]` 在 Bank 0, `M[2]` 在 Bank 1...）。
  - **目的:** 防止所有线程都去挤同一个 Bank，而是将并行线程的内存请求均匀分散到所有 Banks 上。

**Q3: 为什么说 "Maximizing Occupancy" (最大化占用率) 对内存性能至关重要？**

- **A:** 仅仅有 Banks 是不够的，还需要有足够多的**在途内存请求 (In-flight memory requests)** 来填满这些 Banks。
  - 高 Occupancy 意味着有足够多的活跃 Warps。当某些 Warps 在等待 DRAM 数据时，GPU 可以切换到其他 Warps 发起新的请求，从而掩盖 DRAM 的访问延迟。

**Q4: 理论上，一个通道至少需要多少个 Banks 才能跑满带宽？**

- **A:** 这是一个类似 Little's Law 的关系。如果 DRAM 访问延迟是数据传输时间的 $R$ 倍，那么至少需要 $R+1$ 个 Banks 才能完全掩盖延迟。实际上为了避免 Bank Conflict，通常需要更多。

------

### 6.3 Thread Coarsening (线程粗化)

**Q1: 什么是 Thread Coarsening？它与我们之前追求的“最大化并行”是否矛盾？**

- **A:** 它是指让每个线程负责计算**多个**输出元素（例如 2 个或 4 个），而不是 1 个。
  - 这确实减少了并行度（Block 数量变少），但目的是为了消除**并行带来的代价 (Price of Parallelism)**。

**Q2: 在 Tiled 矩阵乘法中，Coarsening 的主要收益是什么？**

- **A:** **减少了 Global Memory 的冗余加载。**
  - 如果不粗化，计算相邻两个 Tile 的两个 Block 需要分别加载相同的矩阵 M 数据。
  - 如果粗化（一个 Block 算两个 Tile），矩阵 M 的数据只需加载一次到 Shared Memory，就可以被复用于计算多个输出位置，从而节省了一半的 M 矩阵带宽。

**Q3: Coarsening 的 "代价" 或 "陷阱" 是什么？（至少列举两点）**

- **A:**
  1. **寄存器压力增大:** 每个线程需要更多的寄存器来保存中间结果（如 `Pvalue[]`），这可能导致 Occupancy 下降。
  2. **硬件利用率不足 (Underutilization):** 如果粗化过度，Block 数量太少，可能无法填满 GPU 所有的 SM，导致硬件空闲。

------

### 6.4 Checklist of Optimizations (优化清单)

**Q1: 请简述 Table 6.1 中的六大优化策略及其主要收益。**

- **A:**
  1. **Maximizing Occupancy:** 掩盖计算流水线延迟 + 掩盖 DRAM 延迟。
  2. **Coalesced Access:** 减少 Global Memory 流量，利用 DRAM Burst。
  3. **Minimizing Control Divergence:** 提高 SIMD 执行效率。
  4. **Tiling:** 利用 Shared Memory 复用数据，减少 Global Memory 流量。
  5. **Privatization (私有化):** 在私有副本上更新，减少原子操作的竞争 (Contention)。
  6. **Thread Coarsening:** 减少冗余工作和冗余内存访问。

**Q2: 什么是 Privatization (私有化)？它主要解决什么问题？**

- **A:** 当多个线程需要更新同一个公共输出（如直方图计数）时，为了避免严重的原子操作冲突，让线程先在私有副本（Register/Shared Mem）上更新，最后再一次性合并到全局内存。这大大减少了争用。

------

### 6.5 Knowing your bottleneck (识别瓶颈)

**Q1: 为什么说优化本质上是一种 "Resource Trading" (资源交换)？**

- **A:** 优化通常是通过消耗一种资源来缓解另一种资源的压力。例如 Tiling 是消耗 Shared Memory 来缓解 Global Memory Bandwidth。

**Q2: 举一个“反向优化”的例子（即优化反而导致性能下降）。**

- **A:** 如果程序的瓶颈是 **Occupancy**（受限于 Shared Memory 容量），此时如果强行使用 **Tiling**（需要更多 Shared Memory），会导致 Occupancy 进一步降低，从而使性能恶化。

**Q3: 如何科学地找到瓶颈？**

- **A:** 不要瞎猜。使用 Profiling Tools (性能分析工具) 来测量。并且要意识到不同硬件架构的瓶颈可能不同。

------

### 6.6 Exercises

#### 练习 1: 实现 Corner Turning (转角优化)

**题目：** Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.

分析：

这道题要求我们实现 6.1 节提到的“转角优化”。

- **输入：** 矩阵 A (行优先)，矩阵 B (列优先)。
- **目标：** 计算 $C = A \times B$。
- **挑战：** 矩阵 B 是列优先的，如果按常规逻辑横着去读（跨列），会导致非合并访问（Uncoalesced）。
- **策略 (Figure 6.4)：**
  - **加载阶段：** 线程按照**物理连续**的方式（竖着读，即沿着列读）把 B 加载到 Shared Memory。
  - **存储阶段：** 为了计算方便，可以在 Shared Memory 中把它转置存好（或者直接存，计算时再跳跃访问 SRAM）。这里我们采用 **"Coalesced Load (按列读) -> Transposed Store (按行存)"** 的策略，这样计算时就像处理普通矩阵一样了。

**代码实现：**

C++

```
__global__ void matrixMulCornerTurning(float* A, float* B, float* C, int Width) {
    // 假设 BLOCK_SIZE 是 TILE_WIDTH
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // C 的行和列 (逻辑坐标)
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // 1. 加载 A (行优先，常规操作)
        // 线程 (ty, tx) 读取 A[row][ph*TW + tx] -> 合并访问
        As[ty][tx] = A[row * Width + ph * TILE_WIDTH + tx];

        // 2. 加载 B (列优先，Corner Turning 核心！)
        // 原始 B 是列优先存储: B[col][row] (物理上)
        // 我们需要加载的 B 的 Tile 对应于 B 的行: (ph*TW + ty) 到 (ph*TW + ty + TW)
        //                     对应于 B 的列: (bx*TW + tx) 到 ...
        
        // 技巧：交换线程的角色来进行加载
        // 让线程以 "列优先" 的方式去读 Global Memory，确保存取连续
        // 线程读取 B 的物理地址: B_base + (列索引 * Height + 行索引)
        // 这里简化假设 Width = Height
        
        // 我们实际上要加载 B 的逻辑块：行范围 [ph*TW, ph*TW+TW], 列范围 [col, col+TW]
        // 但 B 是列优先，所以物理上 "列" 是连续的。
        // 我们让线程 (ty, tx) 去读: 
        // 行 = ph*TW + tx  (注意这里用了 tx 作为行偏移，这是 Corner Turning 的关键)
        // 列 = bx*TW + ty  (注意这里用了 ty 作为列偏移)
        
        // 物理地址 = 列 * Width + 行 = (bx*TW + ty) * Width + (ph*TW + tx)
        // 这样 tx 变化时 (0->31)，我们访问的是 "同一列的连续行"，在列优先矩阵中这是连续的！
        
        // 存入 Bs: 转置存入，方便后面计算
        Bs[tx][ty] = B[(bx * TILE_WIDTH + ty) * Width + (ph * TILE_WIDTH + tx)];

        __syncthreads();

        // 3. 计算 (从 Shared Memory 读，都是 SRAM，无所谓顺序)
        for (int k = 0; k < TILE_WIDTH; ++k) {
            // As是行优先，Bs由于我们转置存入了，现在也是行优先逻辑
            Pvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    C[row * Width + col] = Pvalue;
}
```

------

#### 练习 2: Tiled Matrix Mul 的 Block Size

**题目：** For tiled matrix multiplication, ... for what values of BLOCK_SIZE will the kernel completely avoid uncoalesced accesses to global memory?

**分析：**

- **合并访问 (Coalescing) 的条件：** 一个 Warp (32个线程) 访问的起始地址必须是 32 的倍数（对于某些架构是 128 字节对齐），且线程访问的地址必须是连续的。
- **Tiled 矩阵乘法加载逻辑：**
  - 加载 A：`A[row * Width + (ph * TILE + tx)]`。`row` 固定，`tx` 连续变化。只要 `Width` 是 Warp Size (32) 的倍数，通常都能合并。
  - 加载 B：`B[(ph * TILE + ty) * Width + col]`。`ty` 固定，`col` (即 `bx*TILE + tx`) 连续变化。
- **关键约束：** 每一个 Warp 里的线程 (tx=0..31) 必须都处于**同一行**中。
  - 如果 `BLOCK_SIZE` (即 `TILE_WIDTH`) **小于 32**（比如 16）：
    - 一个 Warp (32线程) 会包含两行线程 (Row 0 的 0-15, Row 1 的 0-15)。
    - 对于矩阵 B 的加载，Row 0 的线程在读第 K 行，Row 1 的线程在读第 K+1 行。
    - 这会导致跨行访问，破坏合并性（虽然在现代 GPU 上可能只是分裂成两个 Transaction，不算完全 Uncoalesced，但严格来说不是完美的单一合并）。
  - **因此，BLOCK_SIZE 必须是 32 的倍数**。
    - 通常我们取 **32**。
    - 如果是 16，则无法形成完美的单次 128-byte Transaction（需要两次 64-byte）。

**答案：** `BLOCK_SIZE` 应该是 **32 的倍数** (e.g., 32)。

------

#### 练习 3: 代码审计 (Coalescing Check)

**题目：** 分析 `foo_kernel` 的内存访问模式。

逐项分析：

(注意：i = blockIdx.x * blockDim.x + threadIdx.x，即全局线性索引)

- **a. Line 05: `a[i]`**
  - `i` 是连续的 (`tid` 连续变化)。`a` 是 Global Memory。
  - **答案：Coalesced (合并)**。
- **b. Line 05: `a_s[threadIdx.x]`**
  - `a_s` 是 `__shared__`。
  - **答案：Not Applicable (不适用)**。Coalescing 是 Global Memory 的概念，Shared Memory 只有 Bank Conflict 的概念。
- **c. Line 07: `b[j\*blockDim.x\*gridDim.x + i]`**
  - `i` 随 `tid` 连续变化。`j` 是循环常数。
  - 这本质上是 `b[Constant + i]`。地址连续。
  - **答案：Coalesced (合并)**。
- **d. Line 07: `c[i\*4 + j]`**
  - `i` 随 `tid` 变化。
  - `Thread 0` 访问 `0 + j`。
  - `Thread 1` 访问 `4 + j`。
  - `Thread 2` 访问 `8 + j`。
  - **跨度 (Stride) = 4**。地址不连续。
  - **答案：Uncoalesced (未合并)**。
- **e. Line 07: `bc_s[...]`**
  - Shared Memory。
  - **答案：Not Applicable**。
- **f. Line 10: `a_s[...]`**
  - Shared Memory。
  - **答案：Not Applicable**。
- **g. Line 10: `d[i + 8]`**
  - `i` 连续。
  - **答案：Coalesced (合并)**。
- **h. Line 11: `bc_s[...]`**
  - Shared Memory。
  - **答案：Not Applicable**。
- **i. Line 11: `e[i\*8]`**
  - `Thread 0` 访问 `0`。
  - `Thread 1` 访问 `8`。
  - **跨度 (Stride) = 8**。严重不连续。
  - **答案：Uncoalesced (未合并)**。

------

#### 练习 4: 计算 OP/B (计算访存比)

题目： 计算不同矩阵乘法版本的 Floating Point to Global Memory Access Ratio (OP/B)。

假设矩阵乘法计算量：$2 \times N^3$ (乘法+加法)。

##### a. Simple Kernel (无优化)

- 每个线程计算 1 个元素，做 $N$ 次乘加 ($2N$ Ops)。

- 读取：读 $N$ 个 A，读 $N$ 个 B。写 1 个 C (忽略写回，通常只算读)。

- **Global Memory Reads:** $2N \times 4$ 字节 (float)。

- 计算访存比：

  

  $$\frac{2N \text{ FLOPs}}{8N \text{ Bytes}} = 0.25 \text{ OP/B}$$

- **评价：** 极低，严重受限于带宽。

##### b. Shared Memory Tiling (Tile = 32 x 32)

- 每个 Block 负责 $32 \times 32$ 个元素。

- 总共有 $N/32$ 个 Phase。

- 在每个 Phase 中：

  - Block 整体从 Global Memory 加载 $32 \times 32$ 个 A 元素 + $32 \times 32$ 个 B 元素。
  - Block 内部做 $32 \times 32 \times 32 \times 2$ 次浮点运算 (每个点做 32 次乘加)。

- **计算量 (Per Tile):** $32 \times 32 \times 32 \times 2 = 65536$ FLOPs。

- **访存量 (Per Tile):** $(32 \times 32 + 32 \times 32) \times 4 \text{ Bytes} = 8192$ Bytes。

- 计算访存比：

  

  $$\frac{65536}{8192} = 8.0 \text{ OP/B}$$

- **评价：** 提升了 32 倍！这就是 Tiling 的威力。实际上 OP/B 提升比例大致等于 Tile Width。

##### c. Thread Coarsening (Tile = 32 x 32, Factor = 4)

- 现在一个线程算 4 个元素。Block 依然负责 $32 \times 32$ 的输出区域（但线程数减少为 $32 \times 8$ 或者 Block 变大，这里假设 Coarsening 指的是复用 M 的那种）。

- 根据 Figure 6.13 的逻辑 (Reuse M)：

  - 处理 1 个 M 的 Tile ($32 \times 32$) 和 **4 个** N 的 Tile ($4 \times 32 \times 32$)。
  - 计算出 **4 个** P 的 Tile ($4 \times 32 \times 32$)。

- **计算量：** $4 \times (32 \times 32 \times 32 \times 2) = 262,144$ FLOPs。

- **访存量：**

  - M 加载 **1 次**：$32 \times 32 \times 4$ Bytes。
  - N 加载 **4 次**：$4 \times (32 \times 32 \times 4)$ Bytes。
  - 总字节：$(1024 + 4096) \times 4 = 20,480$ Bytes。

- 计算访存比：

  

  $$\frac{262144}{20480} \approx 12.8 \text{ OP/B}$$

- **评价：** 比单纯 Tiling 又提升了 60%。因为 M 的加载被分摊了。