#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// 宏定义：检查 CUDA 错误
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// ============================================================================
// PART 1: 辅助函数 - 核心数估算 (直接使用，无需修改)
// ============================================================================
int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct { int SM; int Cores; } sSMtoCores;
    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x50, 128}, {0x60, 64}, {0x70, 64}, {0x75, 64},
        {0x80, 64}, {0x86, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}
    };
    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) 
            return nGpuArchCoresPerSM[index].Cores;
        index++;
    }
    return 0;
}

// ============================================================================
// PART 2: Kernel - 同步机制实验 (对应 4.3 节)
// 任务：实现一个简单的数组移位 (每个元素 = 自己 + 右边邻居)
// ============================================================================
__global__ void shift_kernel(int* d_in, int* d_out, int N) {
    // 静态分配共享内存
    __shared__ int s_data[256]; 

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1. 加载数据到 Shared Memory
    if (idx < N) {
        s_data[tid] = d_in[idx];
    }
    
    // ---------------------------------------------------------
    // TODO 1: 在这里添加必要的同步指令
    // 思考：如果不加，步骤 2 读取 s_data[tid+1] 时，邻居的数据可能还没写进去！
    // ---------------------------------------------------------
    // [在这里填入代码]
    __syncthreads();

    // 2. 计算 (当前值 + 右邻居的值)
    if (idx < N - 1 && tid < blockDim.x - 1) {
        d_out[idx] = s_data[tid] + s_data[tid + 1];
    }
}

// ============================================================================
// PART 3: Kernel - Warp 发散实验 (对应 4.5 节)
// 任务：构造导致 Warp 发散的条件
// ============================================================================
__global__ void divergence_kernel(float* d_data) {
    int tid = threadIdx.x;
    float val = 0.0f;

    // ---------------------------------------------------------
    // TODO 2: 编写一个会导致 Severe Divergence (严重发散) 的条件
    // 提示：使用 tid % 2 相关逻辑，让奇偶线程走不同路径
    // ---------------------------------------------------------
    if ( /* 填入条件 */ tid % 2 == 0 ) {
        val = 1.0f; // 路径 A (快)
    } else {
        val = 2.0f; // 路径 B (快)
    }

    // ---------------------------------------------------------
    // TODO 3: 编写一个不会导致发散 (或者发散最小) 的条件
    // 提示：让同一个 Warp 内的线程 (tid / 32) 走相同的路径
    // ---------------------------------------------------------
    if ( /* 填入条件 */ (tid / 32) % 2 == 0) {
        val += 10.0f;
    }

    d_data[tid] = val;
}

// ============================================================================
// PART 4: Host 函数 - 占用率计算器 (对应 4.7 节)
// 任务：模拟计算 SM 能跑多少个 Block
// ============================================================================
void calculate_occupancy(cudaDeviceProp prop, int blockSize, int regsPerThread) {
    printf("\n--- Occupancy Calculator (BlockSize=%d, Regs/Thread=%d) ---\n", 
           blockSize, regsPerThread);

    // 硬件限制 (分母)
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor; // 例如 2048
    int maxBlocksPerSM  = 16; // 假设架构限制 (不同架构不同，这里简化设为16以便练习)
    int maxRegsPerSM    = prop.regsPerMultiprocessor;     // 例如 65536

    // ---------------------------------------------------------
    // TODO 4: 计算受限于“线程槽” (Thread Slots) 能跑多少个 Block？
    // 公式：SM最大线程数 / Block大小
    // ---------------------------------------------------------
    int limit_by_threads = maxThreadsPerSM / maxBlocksPerSM ; // 修改这里

    // ---------------------------------------------------------
    // TODO 5: 计算受限于“寄存器” (Registers) 能跑多少个 Block？
    // 提示：
    // 1. 一个 Block 需要的总寄存器 = blockSize * regsPerThread
    // 2. SM 能装多少个这样的 Block = maxRegsPerSM / Block总寄存器
    // ---------------------------------------------------------
    int limit_by_regs = maxRegsPerSM / (maxBlocksPerSM * (maxRegsPerSM / maxThreadsPerSM)); // 修改这里

    // 计算最终能跑的 Block 数 (取三者最小值: 线程限制, 寄存器限制, 硬件Block数限制)
    int active_blocks = limit_by_threads;
    if (limit_by_regs < active_blocks) active_blocks = limit_by_regs;
    if (maxBlocksPerSM < active_blocks) active_blocks = maxBlocksPerSM;

    // 计算最终占用率
    int active_threads = active_blocks * blockSize;
    float occupancy = (float)active_threads / maxThreadsPerSM * 100.0f;

    printf("  [Limit Analysis]\n");
    printf("    By Threads: %d blocks\n", limit_by_threads);
    printf("    By Regs:    %d blocks\n", limit_by_regs);
    printf("    By HW Limit:%d blocks\n", maxBlocksPerSM);
    printf("  >> Final Active Blocks: %d\n", active_blocks);
    printf("  >> Occupancy: %.1f%%\n", occupancy);

    if (occupancy < 50.0f) printf("  [!] Warning: Low Occupancy! (Performance Cliff?)\n");
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA device found.\n");
        return -1;
    }

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Device: %s\n", prop.name);
    
    // ---------------------------------------------------------
    // TODO 6: 打印这一章最重要的三个硬件指标
    // 1. SM 数量 (multiProcessorCount)
    // 2. 每个 Block 最大线程数 (maxThreadsPerBlock)
    // 3. 每个 SM 最大寄存器数 (regsPerMultiprocessor)
    // ---------------------------------------------------------
    printf("SM Count: %d\n", prop.multiProcessorCount); // 修改
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock); // 修改
    printf("Max Regs per SM: %d\n", prop.regsPerMultiprocessor); // 修改

    // === 测试 1: 占用率计算 ===
    // 场景 A: 理想情况 (512线程, 32个寄存器)
    calculate_occupancy(prop, 512, 32);
    // 场景 B: 性能悬崖 (512线程, 64个寄存器 - 寄存器用量翻倍)
    calculate_occupancy(prop, 512, 64);

    // === 测试 2: 同步 Kernel ===
    int N = 256;
    int *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CHECK(cudaMalloc(&d_out, N * sizeof(int)));
    
    // 这里的 Grid/Block 随便设个简单的，重点在 Kernel 内部
    shift_kernel<<<1, 256>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    
    printf("\nKernels executed successfully.\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}