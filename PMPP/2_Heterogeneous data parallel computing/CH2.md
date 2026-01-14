# CH2 Heterogeneous data parallel computing



### **2.1 Data Parallelism (数据并行)**

**Q1: 什么是“数据并行” (Data Parallelism)？它与“任务并行”有何不同？**

- **A:**
  - **定义:** 指针对数据集的不同部分进行的计算可以**相互独立**地进行。例如将 100 万个像素点分别进行灰度转换，每个像素的计算不依赖于其他像素。
  - **区别:** 任务并行 (Task Parallelism) 是指同时做不同的任务（如一边放音乐一边下文件）；数据并行是同时做相同的任务但处理不同的数据。

**Q2: 为什么图像处理（如彩色转灰度）是数据并行的绝佳例子？**

- **A:** 因为图像本质上是一个像素数组。计算一个像素的灰度值（$L = r*0.21 + g*0.72 + b*0.07$）只需要该像素自己的 RGB 值，完全不需要知道邻居像素的值。这种**独立性**允许我们同时启动数百万个线程进行处理。

------

### **2.2 CUDA C Program Structure (CUDA C 程序结构)**

**Q3: 一个完整的 CUDA 程序由哪两部分组成？**

- **A:**
  - **Host Code (主机代码):** 运行在 CPU 上，负责逻辑控制、内存分配和数据传输。
  - **Device Code (设备代码/Kernel):** 运行在 GPU 上，负责大规模并行计算。

**Q4: 什么是 Grid (网格)？它与 Kernel (内核) 有什么关系？**

- **A:** Kernel 是代码（函数定义），而 Grid 是运行时的实例。当 Host 调用一次 Kernel 时，GPU 上会启动一大群线程来执行这个 Kernel，这群线程的集合就叫 Grid。

------

### **2.3 A Vector Addition Kernel (向量加法内核)**

**Q5: 在 CUDA 编程中，为什么推荐使用 `_h` 和 `_d` 作为变量后缀？**

- **A:** 这是一个良好的编程习惯。`_h` 表示 Host (CPU) 内存上的指针，`_d` 表示 Device (GPU) 显存上的指针。这能防止程序员在 Host 代码中错误地解引用 Device 指针（这会导致程序崩溃）。

**Q6: 典型的 CUDA “外包模式” (Outsourcing) 分为哪三步？**

- **A:**
  1. **数据传入 (Part 1):** 将数据从 Host 内存拷贝到 Device 显存。
  2. **计算 (Part 2):** 启动 Kernel，GPU 进行并行计算。
  3. **结果传出 (Part 3):** 将结果从 Device 显存拷贝回 Host 内存。

------

### **2.4 Device Global Memory & Data Transfer (显存与数据传输)**

**Q7: `cudaMalloc` 的参数为什么是 `(void**)`？它与 C 语言的 `malloc` 有何不同？**

- **A:** C 语言的 `malloc` 是通过**返回值**返回指针。而 `cudaMalloc` 的返回值被用来报告错误码（`cudaError_t`），所以它必须通过**修改参数**的方式把分配好的地址传出来。因此我们需要传入“指针的指针” `(void**)`。

**Q8: `cudaMemcpy` 的第四个参数有什么作用？**

- **A:** 它指定了数据拷贝的方向。最常用的有 `cudaMemcpyHostToDevice` (上传) 和 `cudaMemcpyDeviceToHost` (下载)。如果填错了方向，会导致严重的运行时错误。

------

### **2.5 Kernel Functions and Threading (内核函数与线程)**

**Q9: 什么是 SPMD (Single-Program Multiple-Data)？**

- **A:** 单程序多数据。这是 CUDA 的编程模型，意味着你只写一个函数（Kernel），但系统会启动成千上万个线程，每个线程都执行这同一段代码，但处理不同的数据块。

**Q10: 如何计算线程的全局索引（Global Index）？（必考公式）**

- **A:** `i = blockIdx.x * blockDim.x + threadIdx.x`

  - `blockIdx.x`: 我在第几个班（Block）？

  - `blockDim.x`: 一个班有多少人？

  - threadIdx.x: 我是班里的第几号？

    这个公式将层级的线程结构映射到了线性的数组索引上。

**Q11: 为什么 Kernel 代码中通常需要 `if (i < n)` 这样的边界检查？**

- **A:** 因为线程块的大小 (`blockDim`) 通常是固定的（如 256 或 512）。当数据总量 $N$ 不是块大小的整数倍时，我们启动的线程总数会略多于 $N$。为了防止多出来的线程访问数组越界，必须让它们“闭嘴”不干活。

**Q12: `__global__`, `__device__`, `__host__` 分别代表什么？**

- **A:**
  - `__global__`: Host 调用，Device 执行（这就是 Kernel）。
  - `__device__`: Device 调用，Device 执行（辅助函数）。
  - `__host__`: Host 调用，Host 执行（普通 C 函数）。

------

### **2.6 Calling Kernel Functions (调用内核函数)**

**Q13: 解释 Kernel 启动语法 `<<<A, B>>>` 中的 A 和 B。**

- **A:**
  - **A (Grid Dimension):** 启动多少个 Block？
  - **B (Block Dimension):** 每个 Block 包含多少个 Thread？。

**Q14: 如果向量长度为 $N$，Block 大小为 256，应该启动多少个 Block？**

- **A:** 应该使用向上取整除法：`ceil(N / 256.0)`。这保证了即使 $N$ 不能被整除，也有足够的线程覆盖所有元素。

**Q15: 什么是“透明的可扩展性” (Transparent Scalability)？**

- **A:** 指同一段 CUDA 代码可以在不同性能的 GPU 上运行。硬件调度器会根据 GPU 的核心数量，自动决定是一次性并行执行所有 Block（大显卡），还是分批次串行执行 Block（小显卡）。程序员不需要修改代码。

------

### **2.7 Compilation (编译)**

**Q16: NVCC 编译器的主要工作流程是什么？**

- **A:** NVCC 充当一个“驱动程序”，它将源代码分离（Separate）：
  - **Host 代码**发送给系统的标准 C++ 编译器（如 GCC/CL）。
  - **Device 代码**由 NVCC 编译成 **PTX**（虚拟汇编）。
  - 最终运行时，PTX 会被 JIT（即时编译）成针对具体硬件的机器码。

------

### **2.8 Summary (总结)**

**Q17: 总结一下，编写一个 CUDA 并行程序需要哪几个关键步骤？**

- **A:**
  1. **cudaMalloc:** 在 GPU 上申请地盘。
  2. **cudaMemcpy (H2D):** 把数据搬进 GPU。
  3. **Kernel Launch:** 写好 `__global__` 函数，并用 `<<<...>>>` 启动。
  4. **cudaMemcpy (D2H):** 把结果搬回 CPU。
  5. **cudaFree:** 释放 GPU 内存。