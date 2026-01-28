# CH7 Convolution

### 7.1 Background (背景知识)

**Q1: 什么是卷积运算的核心逻辑？**

- **A:** 卷积是一种数组运算。每一个**输出元素**都是对应的**输入元素**及其周围**邻居元素**的**加权和 (Weighted Sum)**。权重的集合被称为**滤波器 (Filter)** 或掩码 (Mask)。

**Q2: 什么是 "Ghost Cells" (幽灵单元)？通常如何处理？**

- **A:** 当计算图像边缘的像素时，滤波器会延伸到输入图像的边界之外。那些缺失的、不存在的输入元素被称为 **Ghost Cells**。
- **处理策略:** 最常见的方法是**补零 (Zero Padding)**，即假设这些位置的值为 0。也可以采用复制边界值或镜像反射等方法。

**Q3: 在描述卷积时，为什么书中坚持用 "Filter" 而不是 "Kernel"？**

- **A:** 为了避免术语混淆。在 CUDA 编程中，"Kernel" 特指在 GPU 上运行的函数 (Global Function)；而在数学卷积中，权重数组通常也叫 Kernel。为了区分，书中统一称权重数组为 **Convolution Filter**。

------

### 7.2 Parallel Convolution: A Basic Algorithm (基础并行算法)

**Q1: 基础卷积 Kernel 的线程映射策略是什么？**

- **A:** **一个线程负责计算一个输出像素**。线程组织成 2D Grid，Thread `(tx, ty)` 计算 Output `P[row][col]`。

**Q2: 基础版本的性能瓶颈在哪里？请用数据说明。**

- **A:** 瓶颈是 **Global Memory 带宽**。
  - 计算每个像素需要反复从 Global Memory 读取输入 $N$ 和滤波器 $F$。
  - **计算访存比 (Arithmetic Intensity)** 极低，仅为 **0.25 OP/B**（每读取 8 字节数据进行 2 次浮点运算）。

**Q3: 基础版本中的控制发散 (Control Divergence) 是如何产生的？严重吗？**

- **A:** 产生于处理 **Ghost Cells** 的 `if` 分支。边缘线程会跳过计算，而中心线程会执行计算。
  - 对于大图像来说，边缘像素占比很小，所以这种发散的影响通常**微乎其微**。

------

### 7.3 Constant Memory and Caching (常量内存与缓存)

**Q1: 为什么卷积滤波器 (Filter) 非常适合使用 Constant Memory？(列举三个特性)**

- **A:**
  1. **尺寸小 (Small Size):** 半径通常较小，很容易塞进 64KB 的常量内存。
  2. **只读 (Read-Only):** 在 Kernel 运行期间数值不变。
  3. **全员访问 (Uniform Access):** 所有线程在同一时刻都读取相同的滤波器地址（广播效应）。

**Q2: Constant Cache 相比 L1/L2 Cache 有什么特殊的硬件优势？**

- **A:** 它支持 **广播机制 (Broadcasting)**。当一个 Warp 的所有线程访问同一个地址时，Constant Cache 只需读取一次，然后将数据同时广播给所有线程，极大节省了带宽。

**Q3: 使用 Constant Memory 后，计算访存比提升到了多少？**

- **A:** 提升到了 **0.5 OP/B**。因为读取 $F$ 的带宽消耗几乎降为 0，只剩下读取输入 $N$ 的开销。

------

### 7.4 Tiled Convolution with Halo Cells (带光环的瓦片化)

**Q1: 卷积的 Tiling 和矩阵乘法的 Tiling 最大的区别是什么？**

- **A:** 尺寸不匹配。
  - 矩阵乘法：Input Tile 大小 = Output Tile 大小。
  - 卷积：**Input Tile 大小 > Output Tile 大小**。因为计算输出 Tile 需要额外的一圈数据，称为 **Halo Cells (光环单元)**。

**Q2: 在 "Input-Tile-Centric" (以输入为中心) 的策略中，为什么会有线程闲置？**

- **A:** 我们让 Block 大小等于 Input Tile 大小，启动了足够多的线程来加载所有数据（包括 Halo）。但在**计算阶段**，边缘线程对应的输出位置超出了当前 Tile 的有效计算范围（且属于邻居 Block 的责任区），因此这些线程必须被**关停 (Deactivate)**，导致闲置。

**Q3: 既然有线程闲置，为什么我们还要做 Tiling？**

- **A:** 为了**复用数据**。尽管浪费了线程资源，但我们将 Input Tile 加载到 Shared Memory 后，每个像素被复用了多次。这使得计算访存比从 0.5 OP/B 飙升到 **9.57 OP/B** (针对 5x5 滤波器)，大幅突破了带宽瓶颈。

------

### 7.5 Tiled Convolution Using Caches (利用缓存处理光环)

**Q1: 7.5 节的优化思路是基于什么观察？**

- **A:** 作者观察到，一个 Block 需要的 **Halo Cells**，其实就是邻居 Block 的 **内部数据**。邻居 Block 在运行时很可能已经把这些数据拉进了 **L2 Cache**。因此，我们可以直接去 Global Memory (L2) 读 Halo，而不需要手动搬进 Shared Memory。

**Q2: 相比 7.4 节，7.5 节的代码策略做了哪些简化和妥协？**

- **A:**
  - **简化:** Block 大小 = Output Tile 大小。线程一一对应，不再需要复杂的 Halo 加载逻辑，也**不再有线程闲置**。
  - **妥协 (复杂化):** 计算阶段变复杂了。需要用 `if-else` 判断数据是在 Shared Memory (内部数据) 还是在 Global Memory (Halo 数据)。

**Q3: 为什么说 7.5 节的方法减少了 Memory Divergence？**

- **A:** 在 7.4 节中，Input Tile 和 Output Tile 大小不一致（通常不是 2 的幂），这可能导致内存对齐问题。而 7.5 节可以让 Input/Output/Block 大小保持一致且为 2 的幂（如 32x32），内存访问更规整。

------

### 综合分析与实战判断

**Q1: 如果滤波器尺寸变大（例如从 5x5 变成 9x9），Tiling 的计算访存比 (OP/B) 会变高还是变低？为什么？**

- **A:** 会**变高**。
  - 因为每个输入像素被更多的线程（滤波器里的更多点）复用了。
  - 例如 9x9 滤波器的理论 OP/B 远高于 5x5 滤波器（前提是 Tile 足够大）。

**Q2: 如果 Output Tile 设置得太小（例如 8x8），会有什么后果？**

- **A:** 性能会很差。因为 Halo Cells 的占比会变得非常大（Halo Overhead）。加载了一大堆 Halo 数据，结果只算了中间一点点输出，数据复用率低，OP/B 下降严重。

**Q3: 什么时候该用 7.4 (Shared Mem Halo)，什么时候该用 7.5 (L2 Cache Halo)？**

- **A:**
  - **7.4 (纯 Shared Mem):** 当你需要极致的确定性性能，且 Shared Memory 资源充足时。或者当你处理的数据极度复用，L2 Cache 抖动严重时。
  - **7.5 (L2 Cache):** 现代 GPU 编程的首选。代码更简单，没有线程闲置，且现代 GPU 的 L2 Cache 很大很快，足以应付 Halo 访问。

### 笔试与分析题

**1. Calculate the P[0] value in Fig. 7.3.**

**分析：**

根据 Figure 7.3，输入数组 $x = [8, 2, 5, 4, 1, 7, 3]$，滤波器 $f = [1, 3, 5, 3, 1]$。

计算 $P[0]$ (即 $y[0]$) 时，滤波器中心对准 $x[0]$。

- **输入窗口：** 需要 $x[-2], x[-1], x[0], x[1], x[2]$。
- **Ghost Cells：** $x[-2]$ 和 $x[-1]$ 越界，补 0。
- **有效值：** $0, 0, 8, 2, 5$。

**计算：**

$$P[0] = (1 \times 0) + (3 \times 0) + (5 \times 8) + (3 \times 2) + (1 \times 5)$$

$$P[0] = 0 + 0 + 40 + 6 + 5 = \mathbf{51}$$

**2. 1D Convolution Calculation**

**题目：** Input $N = \{4,1,3,2,3\}$, Filter $F = \{2,1,4\}$.

**分析：** Filter 长度为 3，半径 $r=1$。公式为 $Output[i] = F[0]N[i-1] + F[1]N[i] + F[2]N[i+1]$。假设边界补 0。

- **P[0] (Center 4):** $2(0) + 1(4) + 4(1) = 0 + 4 + 4 = \mathbf{8}$
- **P[1] (Center 1):** $2(4) + 1(1) + 4(3) = 8 + 1 + 12 = \mathbf{21}$
- **P[2] (Center 3):** $2(1) + 1(3) + 4(2) = 2 + 3 + 8 = \mathbf{13}$
- **P[3] (Center 2):** $2(3) + 1(2) + 4(3) = 6 + 2 + 12 = \mathbf{20}$
- **P[4] (Center 3):** $2(2) + 1(3) + 4(0) = 4 + 3 + 0 = \mathbf{7}$

**结果：** Output Array = **{8, 21, 13, 20, 7}**

**3. Filter Analysis**

**分析各滤波器对信号的作用：**

- **a. [0 1 0]:** **Identity (恒等变换)**。输出等于输入，不做任何改变。
- **b. [0 0 1]:** **Shift Left (左移)**。当前位置的值取自右边的邻居 ($x[i+1]$)，导致整个信号向左移动一格。
- **c. [1 0 0]:** **Shift Right (右移)**。当前位置的值取自左边的邻居 ($x[i-1]$)，导致整个信号向右移动一格。
- **d. [-1/2 0 1/2]:** (假设题意为常见的微分算子)。这是 **Derivative/Gradient (一阶导数)** 滤波器，用于检测边缘。它计算右邻居与左邻居的差值 (Central Difference)。
- **e. [1/3 1/3 1/3]:** **Smoothing/Box Blur (平滑/模糊)**。取当前及左右邻居的平均值，消除噪声，平滑信号。

**4. 1D Convolution Complexity (Size N, Filter M)**

设半径 $r = (M-1)/2$。

- **a. Total Ghost Cells (unique locations):** 左边缺 $r$ 个，右边缺 $r$ 个。总共 **$2r = M-1$** 个幽灵位置。
- **b. Multiplications (Treated as 0):** 每个输出点都做完整卷积。总运算量 = **$N \times M$**。
- **c. Multiplications (Untreated/Skipped):**
  - 总运算量 - 涉及 Ghost 的运算量。
  - Ghost 访问次数 = 左边界 $r(r+1)/2$ + 右边界 $r(r+1)/2 = r(r+1)$。
  - 有效运算量 = **$N \times M - r(r+1)$**。

5. **2D Convolution Complexity (Square $N \times N$, Filter $M \times M$)**

设半径 $r = (M-1)/2$。

- **a. Total Ghost Cells (locations):** 扩充一圈 $r$ 后的面积减去原面积。$(N+2r)^2 - N^2 = \mathbf{4Nr + 4r^2}$。
- **b. Multiplications (Treated):** **$N^2 \times M^2$**。
- **c. Multiplications (Untreated):**
  - 我们在 1D 中推导出的“有效乘法数”是 $N \times M - r(r+1)$。
  - 2D 卷积是可分离的 (对于计数来说)，有效计算总是发生在“有效行”和“有效列”的交集。
  - 公式为：**$(N \times M - r(r+1))^2$**。即 x 方向有效乘法数乘以 y 方向有效乘法数。

**6. 2D Rectangular Complexity ($N_1 \times N_2$, Filter $M_1 \times M_2$)**

设 $r_1, r_2$ 分别为 x, y 方向半径。

- **a. Total Ghost Cells:** **$(N_1+2r_1)(N_2+2r_2) - N_1 N_2$**。
- **b. Multiplications (Treated):** **$(N_1 N_2) \times (M_1 M_2)$**。
- **c. Multiplications (Untreated):** **$(N_1 M_1 - r_1(r_1+1)) \times (N_2 M_2 - r_2(r_2+1))$**。

**7. Tiled 2D Convolution Analysis**

参数：Array $N \times N$, Filter $M \times M$ ($r=(M-1)/2$), Output Tile $T \times T$。

- **Case 1: Using Kernel in Fig 7.12 (Block Size = Input Tile Size)**
  - **a. Thread Blocks:** 覆盖 $N \times N$ 输出，每块负责 $T \times T$。需要 **$(\lceil N/T \rceil)^2$** 个 Block。
  - **b. Threads per Block:** Block 大小等于 Input Tile 大小。Input Tile 边长为 $T + 2r$。线程数 = **$(T + M - 1)^2$**。
  - **c. Shared Memory:** 存储 Input Tile。需要 **$(T + M - 1)^2 \times 4$ bytes** (assuming float)。
- **Case 2: Using Kernel in Fig 7.15 (Block Size = Output Tile Size, L2 Cache for Halo)**
  - **a. Thread Blocks:** 同样是 **$(\lceil N/T \rceil)^2$**。
  - **b. Threads per Block:** Block 大小等于 Output Tile 大小。线程数 = **$T^2$**。
  - **c. Shared Memory:** 只存储 Internal Elements。需要 **$T^2 \times 4$ bytes**。

------

### 编程与实现题 (3D Convolution)

3D 卷积在视频处理和医学影像（如 CT/MRI）中非常常见。输入是 $(x, y, z)$，滤波器是 $(f_x, f_y, f_z)$。

**8. Revise Basic Kernel (Fig 7.7) for 3D**

**思路：** 增加 `z` 维度。线程索引变为 3D，循环变为 3 层嵌套。

```c++
__global__ void convolution_3D_basic(float *N, float *F, float *P, 
                                     int r, int width, int height, int depth) {
    // 1. 计算 3D 坐标
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outSlice = blockIdx.z * blockDim.z + threadIdx.z; // 新增深度索引

    float Pvalue = 0.0f;
    
    // 2. 三层循环遍历 3D 滤波器
    for (int fSlice = 0; fSlice < 2*r+1; fSlice++) {     // Z 轴
        for (int fRow = 0; fRow < 2*r+1; fRow++) {       // Y 轴
            for (int fCol = 0; fCol < 2*r+1; fCol++) {   // X 轴
                
                int inSlice = outSlice - r + fSlice;
                int inRow   = outRow   - r + fRow;
                int inCol   = outCol   - r + fCol;

                // 3. 3D 边界检查
                if (inCol >= 0 && inCol < width && 
                    inRow >= 0 && inRow < height && 
                    inSlice >= 0 && inSlice < depth) {
                    
                    // 线性索引计算：z * (H*W) + y * W + x
                    int fIdx = fSlice*(2*r+1)*(2*r+1) + fRow*(2*r+1) + fCol;
                    int nIdx = inSlice*(width*height) + inRow*width + inCol;
                    
                    Pvalue += F[fIdx] * N[nIdx];
                }
            }
        }
    }
    
    // 4. 写回输出
    if (outCol < width && outRow < height && outSlice < depth) {
        P[outSlice*(width*height) + outRow*width + outCol] = Pvalue;
    }
}
```

**9. Revise Constant Memory Kernel (Fig 7.9) for 3D**

**思路：** 在 Host 端声明 3D `__constant__` 数组。Kernel 中直接访问 `F[z][y][x]`。

```c++
// Host 端声明 (假设 r=2, 大小 5x5x5)
__constant__ float F_3D[5][5][5];

__global__ void convolution_3D_constant(float *N, float *P, 
                                        int r, int width, int height, int depth) {
    // ... (坐标计算同上) ...
    
    // 核心循环：直接访问 F_3D
    for (int fSlice = 0; fSlice < 2*r+1; fSlice++) {
        for (int fRow = 0; fRow < 2*r+1; fRow++) {
            for (int fCol = 0; fCol < 2*r+1; fCol++) {
                
                // ... (坐标计算与边界检查同上) ...
                
                if (isValid) {
                    // 使用常量内存 F_3D[z][y][x]
                    Pvalue += F_3D[fSlice][fRow][fCol] * N[nIdx];
                }
            }
        }
    }
    // ...
}
```

**10. Revise Tiled Kernel (Fig 7.12) for 3D**

**思路：** 这是最具挑战性的。我们需要将 3D Input Tile 加载到 3D Shared Memory 中。Block 大小等于 Input Tile 大小。

```c++
// 假设 BLOCK_SIZE 是 Input Tile 的维度 (包含 Halo)
#define IN_TILE_DIM 10 // 举例: 8(output) + 2(halo)

__global__ void convolution_3D_tiled(float *N, float *P, 
                                     int r, int width, int height, int depth) {
    // 1. 计算输出 Tile 的基准位置
    int out_tile_w = IN_TILE_DIM - 2*r;
    int col_start = blockIdx.x * out_tile_w;
    int row_start = blockIdx.y * out_tile_w;
    int slice_start = blockIdx.z * out_tile_w;

    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

    // 当前线程对应的全局坐标 (用于加载 Input Tile)
    int col = col_start + tx - r;
    int row = row_start + ty - r;
    int slice = slice_start + tz - r;

    // 2. 声明 3D Shared Memory
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // 3. 加载数据 (处理 Ghost Cells)
    if (col >= 0 && col < width && row >= 0 && row < height && slice >= 0 && slice < depth) {
        N_s[tz][ty][tx] = N[slice*width*height + row*width + col];
    } else {
        N_s[tz][ty][tx] = 0.0f;
    }
    __syncthreads();

    // 4. 计算输出 (关停 Halo 线程)
    int tileCol = tx - r;
    int tileRow = ty - r;
    int tileSlice = tz - r;

    // 只有在 Output Tile 范围内的线程才计算
    if (tileCol >= 0 && tileCol < out_tile_w &&
        tileRow >= 0 && tileRow < out_tile_w &&
        tileSlice >= 0 && tileSlice < out_tile_w) {
        
        // 全局边界检查
        if (col_start + tileCol < width && 
            row_start + tileRow < height && 
            slice_start + tileSlice < depth) {
            
            float Pvalue = 0.0f;
            // 3D 卷积计算 (从 Shared Memory 读取)
            for (int k = 0; k < 2*r+1; k++) {
                for (int i = 0; i < 2*r+1; i++) {
                    for (int j = 0; j < 2*r+1; j++) {
                        // 使用常量内存 F 和 共享内存 N_s
                        Pvalue += F_3D[k][i][j] * N_s[tz + k - r][ty + i - r][tx + j - r];
                        // 注意索引：SharedMem 中心的偏移逻辑需要仔细推导
                        // 这里的简单写法：N_s[tz][ty][tx] 是当前点，邻居直接加偏移
                        // N_s[tz - r + k][ty - r + i][tx - r + j]
                    }
                }
            }
            // 写回 Global Memory
            int outIdx = (slice_start+tileSlice)*width*height + 
                         (row_start+tileRow)*width + (col_start+tileCol);
            P[outIdx] = Pvalue;
        }
    }
}
```