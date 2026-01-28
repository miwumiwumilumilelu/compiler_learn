#include <stdio.h>
#include <cuda_runtime.h>

// =========================================================
// 参数配置
// =========================================================
#define FILTER_RADIUS 2
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

// Tile 配置
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

// 宏：检查 CUDA 错误
#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

// ---------------------------------------------------------
// 7.3 知识点：声明常量内存
// TODO 1: 在这里声明存放在 Constant Memory 的滤波器数组 F_const
// 大小应该是 [FILTER_DIM][FILTER_DIM]
// ---------------------------------------------------------
// __constant__ float ...



// =========================================================
// 7.1 背景知识：CPU 参考实现
// =========================================================
void convolution_cpu(float *h_N, float *h_F, float *h_P, int width, int height)
{
    for (int outRow = 0; outRow < height; outRow++)
    {
        for (int outCol = 0; outCol < width; outCol++)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < FILTER_DIM; fRow++)
            {
                for (int fCol = 0; fCol < FILTER_DIM; fCol++)
                {
                    int inRow = outRow - FILTER_RADIUS + fRow;
                    int inCol = outCol - FILTER_RADIUS + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    {
                        Pvalue += h_F[fRow * FILTER_DIM + fCol] * h_N[inRow * width + inCol];
                    }
                }
            }
            h_P[outRow * width + outCol] = Pvalue;
        }
    }
}

// =========================================================
// Kernel 1 (7.2 Basic): 基础全局内存版本
// =========================================================
__global__ void convolution_basic(float *N, float *F, float *P, int width, int height)
{
    // 计算输出坐标
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol < width && outRow < height)
    {
        float Pvalue = 0.0f;
        // -----------------------------------------------------
        // TODO 2: 实现基础卷积逻辑
        // 1. 双重循环遍历滤波器
        // 2. 计算对应的输入坐标 (inRow, inCol)
        // 3. 边界检查 (Ghost Cells 处理)
        // 4. 累加 Pvalue
        // -----------------------------------------------------
        
        P[outRow * width + outCol] = Pvalue;
    }
}

// =========================================================
// Kernel 2 (7.3 Constant): 常量内存版本
// =========================================================
__global__ void convolution_constant(float *N, float *P, int width, int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol < width && outRow < height)
    {
        float Pvalue = 0.0f;
        // -----------------------------------------------------
        // TODO 3: 实现常量内存卷积
        // 逻辑与 Basic 版本几乎一样，唯一的区别是：
        // 读取滤波器时，直接使用全局声明的 F_const[fRow][fCol]
        // 而不是从参数 F 里读。
        // -----------------------------------------------------

        P[outRow * width + outCol] = Pvalue;
    }
}

// =========================================================
// Kernel 3 (7.4 Tiled Shared Halo): 共享内存 Halo 版本
// 策略：Block Dim = Input Tile Dim (32x32)
//       Output Tile Dim = 28x28 (会有线程闲置)
// =========================================================
__global__ void convolution_tiled_shared(float *N, float *P, int width, int height)
{
    // -----------------------------------------------------
    // TODO 4: 声明 Shared Memory
    // 大小应该是 [IN_TILE_DIM][IN_TILE_DIM]
    // -----------------------------------------------------
    // __shared__ float N_s...

    // 1. 加载阶段 (协作加载)
    // 计算当前线程负责加载的 Global Memory 坐标
    // 提示：Block 对应 Output Tile 的左上角，但需要向左/上偏移 Radius 才能对齐 Input Tile
    // 这里的逻辑比较绕，参考书本 Figure 7.12
    int tileCol = threadIdx.x; 
    int tileRow = threadIdx.y;
    
    // 计算该 Block 负责的输出区域的左上角 (不含 halo)
    int blockOutCol = blockIdx.x * OUT_TILE_DIM;
    int blockOutRow = blockIdx.y * OUT_TILE_DIM;
    
    // 当前线程对应的真实 Input 坐标 (包含 halo 偏移)
    int inCol = blockOutCol + tileCol - FILTER_RADIUS;
    int inRow = blockOutRow + tileRow - FILTER_RADIUS;

    // -----------------------------------------------------
    // TODO 5: 加载数据到 Shared Memory
    // 检查 inCol, inRow 是否在图像范围内，在则加载，不在则填 0
    // -----------------------------------------------------

    __syncthreads();

    // 2. 计算阶段
    // -----------------------------------------------------
    // TODO 6: 计算输出
    // 1. 只有当 threadIdx 在 [0, IN_TILE_DIM) 范围内，
    //    且属于 Output Tile 区域 (即排除 Halo 线程) 时才计算。
    //    有效范围：tileCol 必须 >= FILTER_RADIUS 且 < OUT_TILE_DIM + FILTER_RADIUS
    // 2. 累加计算 Pvalue (只读 N_s 和 F_const)
    // 3. 写回 Global Memory (注意：写回时坐标是 blockOutCol + tileCol - FILTER_RADIUS)
    // -----------------------------------------------------
    
    // 提示：这部分最容易写错，如果太难，可以先跳过做 Kernel 4
}

// =========================================================
// Kernel 4 (7.5 Tiled L2 Cache): L2 Cache 版本
// 策略：Block Dim = Output Tile Dim (32x32)
//       不加载 Halo 到 Shared Memory，Halo 直接读 Global (L2)
// =========================================================
__global__ void convolution_tiled_L2(float *N, float *P, int width, int height)
{
    // Block 大小现在直接等于 Tile 大小 (例如 16x16 或 32x32)
    const int TILE_SIZE = 32; // 假设 Block 是 32x32
    
    // -----------------------------------------------------
    // TODO 7: 声明 Shared Memory
    // 只需要存内部数据，不需要 Halo，所以大小是 [TILE_SIZE][TILE_SIZE]
    // -----------------------------------------------------
    // __shared__ float N_s...

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outCol = blockIdx.x * TILE_SIZE + tx;
    int outRow = blockIdx.y * TILE_SIZE + ty;

    // 1. 加载内部数据 (Internal Elements)
    if (outCol < width && outRow < height) {
        // TODO 8: 加载 N[outRow][outCol] 到 N_s[ty][tx]
    } else {
        // TODO 8: 填充 0
    }
    __syncthreads();

    // 2. 计算阶段
    if (outCol < width && outRow < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
            for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
                // -----------------------------------------------------
                // TODO 9: 混合数据读取
                // 检查需要的邻居 (tx - r + fCol, ty - r + fRow) 是否在 Shared Memory 范围内
                // 如果在 -> 读 N_s
                // 如果不在 -> 读 N (Global Memory, 蹭 L2 Cache)
                // -----------------------------------------------------
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}


int main() {
    int width = 4096;
    int height = 4096;
    size_t size_bytes = width * height * sizeof(float);
    size_t f_size_bytes = FILTER_DIM * FILTER_DIM * sizeof(float);

    printf("Image: %d x %d\n", width, height);

    // 分配内存
    float *h_N = (float*)malloc(size_bytes);
    float *h_F = (float*)malloc(f_size_bytes);
    float *h_P_cpu = (float*)malloc(size_bytes);
    float *h_P_gpu = (float*)malloc(size_bytes);

    // 初始化
    for(int i=0; i<width*height; ++i) h_N[i] = (float)(rand() % 10);
    for(int i=0; i<FILTER_DIM*FILTER_DIM; ++i) h_F[i] = (float)(rand() % 3);

    float *d_N, *d_P, *d_F;
    CHECK(cudaMalloc(&d_N, size_bytes));
    CHECK(cudaMalloc(&d_P, size_bytes));
    CHECK(cudaMalloc(&d_F, f_size_bytes)); // 给 Kernel 1 用的

    CHECK(cudaMemcpy(d_N, h_N, size_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_F, h_F, f_size_bytes, cudaMemcpyHostToDevice));

    // -----------------------------------------------------
    // TODO 10 (7.3知识点): 将 Host 的 h_F 数据拷贝到 Device 的 Constant Memory (F_const)
    // 使用 cudaMemcpyToSymbol
    // -----------------------------------------------------
    // cudaMemcpyToSymbol(...);

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0;

    // === 1. 运行 CPU (太慢可以注释掉) ===
    // printf("Running CPU...\n");
    // convolution_cpu(h_N, h_F, h_P_cpu, width, height);

    // === 2. 运行 Kernel 1 (Basic) ===
    dim3 block(16, 16);
    dim3 grid((width + 15)/16, (height + 15)/16);
    
    CHECK(cudaMemset(d_P, 0, size_bytes));
    cudaEventRecord(start);
    convolution_basic<<<grid, block>>>(d_N, d_F, d_P, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel 1 (Basic): %f ms\n", ms);

    // === 3. 运行 Kernel 2 (Constant) ===
    CHECK(cudaMemset(d_P, 0, size_bytes));
    cudaEventRecord(start);
    convolution_constant<<<grid, block>>>(d_N, d_P, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel 2 (Constant): %f ms\n", ms);

    // === 4. 运行 Kernel 3 (Tiled Shared) ===
    // 注意：Grid 计算方式变了！因为每个 Block 只产出 OUT_TILE_DIM 大小的结果
    dim3 block3(IN_TILE_DIM, IN_TILE_DIM); // 32x32
    dim3 grid3((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    
    CHECK(cudaMemset(d_P, 0, size_bytes));
    cudaEventRecord(start);
    convolution_tiled_shared<<<grid3, block3>>>(d_N, d_P, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel 3 (Tiled Shared): %f ms\n", ms);

    // === 5. 运行 Kernel 4 (Tiled L2) ===
    // 这里的 Grid 又变回去了，或者用 Tile Size
    dim3 block4(32, 32);
    dim3 grid4((width + 31)/32, (height + 31)/32);
    
    CHECK(cudaMemset(d_P, 0, size_bytes));
    cudaEventRecord(start);
    convolution_tiled_L2<<<grid4, block4>>>(d_N, d_P, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel 4 (Tiled L2): %f ms\n", ms);
    
    // 清理
    cudaFree(d_N); cudaFree(d_P); cudaFree(d_F);
    free(h_N); free(h_F); free(h_P_cpu); free(h_P_gpu);
    return 0;
}