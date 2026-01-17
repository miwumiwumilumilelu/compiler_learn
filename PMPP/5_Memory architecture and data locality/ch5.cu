#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// ============================================================================
// 宏定义与辅助函数
// ============================================================================
#define TILE_WIDTH 16  // 静态 Tiling 使用的块大小

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

void randomInit(float *data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

// ============================================================================
// KERNEL 1: 静态共享内存 Tiling (对应 5.4 和 5.5 节)
// 任务：实现带有边界检查的分块矩阵乘法
// ============================================================================
__global__ void matrixMulStatic(const float *A, const float *B, float *C, 
                                int M, int N, int K) 
{
    // --------------------------------------------------------------------
    // TODO 1: 声明静态共享内存
    // 需要两个二维数组 As 和 Bs，大小均为 [TILE_WIDTH][TILE_WIDTH]
    // --------------------------------------------------------------------
    // [在这里填入代码]

    // 计算索引
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 计算当前线程负责计算 C 中哪个元素的坐标 (row, col)
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // 循环遍历所有的 Tile (Phases)
    // num_phases = ceil(K / TILE_WIDTH)
    int num_phases = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < num_phases; ++ph) {
        
        // ----------------------------------------------------------------
        // TODO 2: 协作加载 A 的 Tile 到 As (包含边界检查)
        // 目标: 加载 A[row, ph*TILE_WIDTH + tx]
        // 提示: 
        // 1. 计算 A 的线性索引。
        // 2. 检查 (row < M) 和 (列索引 < K)。
        // 3. 如果越界，As[ty][tx] 填 0.0f。
        // ----------------------------------------------------------------
        // [在这里填入代码]

        // ----------------------------------------------------------------
        // TODO 3: 协作加载 B 的 Tile 到 Bs (包含边界检查)
        // 目标: 加载 B[ph*TILE_WIDTH + ty, col]
        // 提示: 
        // 1. 计算 B 的线性索引。
        // 2. 检查 (行索引 < K) 和 (col < N)。
        // 3. 如果越界，Bs[ty][tx] 填 0.0f。
        // ----------------------------------------------------------------
        // [在这里填入代码]

        // ----------------------------------------------------------------
        // TODO 4: 第一次同步
        // 思考: 为什么要在这里同步？(Read-After-Write)
        // ----------------------------------------------------------------
        // [在这里填入代码]

        // ----------------------------------------------------------------
        // TODO 5: 计算 Partial Dot Product
        // 在 Shared Memory 上进行矩阵乘法累加
        // ----------------------------------------------------------------
        for (int k = 0; k < TILE_WIDTH; ++k) {
            // [在这里填入代码]
            // Pvalue += ...
        }

        // ----------------------------------------------------------------
        // TODO 6: 第二次同步
        // 思考: 为什么要在这里同步？(Write-After-Read)
        // ----------------------------------------------------------------
        // [在这里填入代码]
    }

    // ----------------------------------------------------------------
    // TODO 7: 写回结果
    // 提示: 只有在 C 矩阵范围内的线程才写回 (row < M && col < N)
    // ----------------------------------------------------------------
    // [在这里填入代码]
}

// ============================================================================
// KERNEL 2: 动态共享内存 Tiling (对应 5.6 节)
// 任务：使用 extern __shared__ 实现，Tile 大小由 Host 端传入
// 注意：动态共享内存是一维数组，需要手动计算索引
// ============================================================================
__global__ void matrixMulDynamic(const float *A, const float *B, float *C, 
                                 int M, int N, int K, int tile_width) 
{
    // ----------------------------------------------------------------
    // TODO 8: 声明动态共享内存
    // 关键字: extern __shared__
    // 类型: float
    // 变量名: s_mem[] (大小未定)
    // ----------------------------------------------------------------
    // [在这里填入代码]

    // ----------------------------------------------------------------
    // TODO 9: 指针偏移
    // s_mem 是一个大的一维数组。
    // 让 float* As 指向 s_mem 的开头。
    // 让 float* Bs 指向 s_mem + (tile_width * tile_width) 的位置。
    // ----------------------------------------------------------------
    // float *As = ...
    // float *Bs = ...

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    float Pvalue = 0.0f;

    int num_phases = (K + tile_width - 1) / tile_width;

    for (int ph = 0; ph < num_phases; ++ph) {
        
        // ------------------------------------------------------------
        // TODO 10: 加载数据 (注意：现在 As 和 Bs 是一维指针)
        // 访问 As[ty][tx] 需要变为 As[ty * tile_width + tx]
        // ------------------------------------------------------------
        
        // Load A
        if (row < M && (ph * tile_width + tx) < K)
            // [在这里填入代码] As[...] = ...
        else
            // [在这里填入代码] As[...] = 0.0f;

        // Load B
        if ((ph * tile_width + ty) < K && col < N)
            // [在这里填入代码] Bs[...] = ...
        else
            // [在这里填入代码] Bs[...] = 0.0f;

        __syncthreads();

        // Compute
        for (int k = 0; k < tile_width; ++k) {
            // [在这里填入代码] Pvalue += As[...] * Bs[...]
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char **argv)
{
    // 故意设定非 16 倍数的大小，强制测试边界检查
    int M = 500; 
    int N = 600; 
    int K = 400; 
    // 动态 Tiling 的大小 (可以是任意值，只要 Shared Memory 够)
    int DYNAMIC_TILE_WIDTH = 16; 

    printf("Matrix Size: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 内存
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_static = (float *)malloc(size_C);
    float *h_C_dynamic = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C);

    randomInit(h_A, M * K);
    randomInit(h_B, K * N);

    // CPU 黄金标准计算 (用于验证)
    printf("Computing CPU Reference...\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    // Device 内存
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, size_A));
    CHECK(cudaMalloc((void **)&d_B, size_B));
    CHECK(cudaMalloc((void **)&d_C, size_C));

    CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // 测试 1: 静态 Shared Memory Kernel
    // ------------------------------------------------------------
    printf("\n>>> Testing Static Shared Memory Kernel <<<\n");
    
    // ------------------------------------------------------------
    // TODO 11: 配置 Grid 和 Block
    // Block: 使用 TILE_WIDTH
    // Grid: 覆盖 M 和 N (注意向上取整)
    // ------------------------------------------------------------
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMulStatic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_C_static, d_C, size_C, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------
    // 测试 2: 动态 Shared Memory Kernel
    // ------------------------------------------------------------
    printf("\n>>> Testing Dynamic Shared Memory Kernel <<<\n");
    
    // 重新计算 Grid (虽然这里一样，但逻辑上可能不同)
    dim3 blockDimDyn(DYNAMIC_TILE_WIDTH, DYNAMIC_TILE_WIDTH);
    dim3 gridDimDyn((N + DYNAMIC_TILE_WIDTH - 1) / DYNAMIC_TILE_WIDTH, 
                    (M + DYNAMIC_TILE_WIDTH - 1) / DYNAMIC_TILE_WIDTH);

    // ------------------------------------------------------------
    // TODO 12: 计算需要的 Shared Memory 字节数
    // 公式: 2 个矩阵 * 宽 * 高 * sizeof(float)
    // ------------------------------------------------------------
    size_t sharedMemSize = 2 * DYNAMIC_TILE_WIDTH * DYNAMIC_TILE_WIDTH * sizeof(float);
    
    // 启动 Kernel (注意第3个参数)
    matrixMulDynamic<<<gridDimDyn, blockDimDyn, sharedMemSize>>>(d_A, d_B, d_C, M, N, K, DYNAMIC_TILE_WIDTH);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_C_dynamic, d_C, size_C, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------
    // 结果验证
    // ------------------------------------------------------------
    printf("\nVerifying results...\n");
    bool passed = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C_ref[i] - h_C_static[i]) > 1e-2) {
            printf("Static Kernel Failed at index %d! CPU=%.4f GPU=%.4f\n", i, h_C_ref[i], h_C_static[i]);
            passed = false;
            break;
        }
        if (fabs(h_C_ref[i] - h_C_dynamic[i]) > 1e-2) {
            printf("Dynamic Kernel Failed at index %d! CPU=%.4f GPU=%.4f\n", i, h_C_ref[i], h_C_dynamic[i]);
            passed = false;
            break;
        }
    }

    if (passed) printf("TEST PASSED! All kernels are correct.\n");

    // 资源清理
    free(h_A); free(h_B); free(h_C_static); free(h_C_dynamic); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}