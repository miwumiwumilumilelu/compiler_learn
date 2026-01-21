#include <stdio.h>
#include <cuda_runtime.h>

// =========================================================
// 配置区域
// =========================================================
// Tile 宽度，对应 Warp Size (32) 以获得最佳性能
#define TILE_DIM 32 

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

// =========================================================
// KERNEL 1: 朴素版转置 (Naive Copy)
// 现象：读取是合并的(Coalesced)，但写入是严重跨步的(Strided/Uncoalesced)。
// =========================================================
__global__ void transposeNaive(float *out, const float *in, int width, int height)
{
    // 计算当前线程对应的全局 2D 坐标 (x, y)
    // x 对应矩阵的列 (width方向), y 对应矩阵的行 (height方向)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (x < width && y < height)
    {
        // --------------------------------------------------------
        // TODO 1: 计算输入和输出的线性索引并进行拷贝
        // 1. 计算输入索引 idx_in：基于 (y, x) 和 width
        // 2. 计算输出索引 idx_out：注意转置后，原来的 (y, x) 变成了 (x, y)，
        //    且输出矩阵的宽度变成了 height。
        // --------------------------------------------------------
        
        // int idx_in = ...
        // int idx_out = ...
        int idx_in = y * width + x;
        int idx_out = x * height + y;
        
        // out[idx_out] = in[idx_in];
        out[idx_out] = in[idx_in];
    }
}

// =========================================================
// KERNEL 2: 优化版转置 (Coalesced + No Bank Conflict)
// 策略：先将数据合并读取到 Shared Memory (Smem)，
//       然后在 Smem 内部进行坐标变换 (转角)，
//       最后合并写入到 Global Memory。
// =========================================================
__global__ void transposeCoalesced(float *out, const float *in, int width, int height)
{
    // --------------------------------------------------------
    // TODO 2: 声明 Shared Memory
    // 思考：为了避免 Bank Conflict，列数需要是一个特殊的值 (Padding)。
    // 如果不加 Padding，列数是 TILE_DIM (32)。
    // 提示：声明一个 [TILE_DIM][TILE_DIM + 1] 的 float 数组。
    // --------------------------------------------------------
    // __shared__ float tile[...][...];
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 计算当前线程在原矩阵中的全局坐标 (xIndex, yIndex)
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    // 计算该线程在原矩阵中的线性索引 (用于读取)
    int index_in = yIndex * width + xIndex;

    // --------------------------------------------------------
    // TODO 3: 读取数据到 Shared Memory (Coalesced Load)
    // 1. 边界检查：确保 xIndex < width && yIndex < height
    // 2. 将 Global Memory 的数据加载到 tile[threadIdx.y][threadIdx.x]
    // 思考：为什么是 [ty][tx]？因为我们要保持 warp 内连续的 tx 访问连续的内存。
    // --------------------------------------------------------
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = in[index_in];
    }
    // if (...) {
    //     tile[...][...] = in[...];
    // }

    // --------------------------------------------------------
    // TODO 4: 线程同步
    // 必须等待 Block 内所有线程都把数据搬进 Shared Memory
    // --------------------------------------------------------
    // ...
    __syncthreads();

    // --- 关键步骤：坐标变换 (Corner Turning) ---
    
    // 我们现在要计算“写入”时的坐标。
    // 在输出矩阵中，原来的 blockIdx.y 变成了新的 x 轴位置
    // 原来的 blockIdx.x 变成了新的 y 轴位置
    int xIndex_new = blockIdx.y * TILE_DIM + threadIdx.x;
    int yIndex_new = blockIdx.x * TILE_DIM + threadIdx.y;

    // 计算输出矩阵的线性索引
    // 注意：输出矩阵的宽度是 height (因为转置了)
    int index_out = yIndex_new * height + xIndex_new;

    // --------------------------------------------------------
    // TODO 5: 将 Shared Memory 数据写入 Global Memory (Coalesced Store)
    // 1. 边界检查：确保 xIndex_new < height && yIndex_new < width
    // 2. 将 tile 中的数据写回 out[index_out]
    // 难点：我们在 Smem 里取数据时，坐标要互换吗？
    // 提示：我们之前存入是 tile[ty][tx]。
    //       现在线程 (tx, ty) 变成了输出矩阵的 (row, col)。
    //       原本的数据应该在 tile 的什么位置？
    //       我们需要读取 tile[threadIdx.x][threadIdx.y] 吗？
    // --------------------------------------------------------
    if (xIndex_new < height && yIndex_new < width) {
        out[index_out] = tile[threadIdx.x][threadIdx.y];
    }
    // if (...) {
    //     out[...] = tile[...][...];
    // }
}

int main()
{
    // 矩阵大小：2048 x 2048 (足够大以体现性能差异)
    const int N = 2048; 
    const int MEM_SIZE = N * N * sizeof(float);

    printf("Matrix Size: %d x %d\n", N, N);

    float *h_in = (float *)malloc(MEM_SIZE);
    float *h_out_naive = (float *)malloc(MEM_SIZE);
    float *h_out_opt = (float *)malloc(MEM_SIZE);
    float *d_in, *d_out;

    // 初始化输入数据
    for (int i = 0; i < N * N; i++) h_in[i] = (float)(rand() % 100);

    CHECK(cudaMalloc(&d_in, MEM_SIZE));
    CHECK(cudaMalloc(&d_out, MEM_SIZE));
    CHECK(cudaMemcpy(d_in, h_in, MEM_SIZE, cudaMemcpyHostToDevice));

    // 配置 Grid 和 Block
    // 确保覆盖整个矩阵
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // 计时变量
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms_naive = 0, ms_opt = 0;

    // --------------------------------------------------------
    // 测试 1: Naive Kernel
    // --------------------------------------------------------
    CHECK(cudaMemset(d_out, 0, MEM_SIZE));
    cudaEventRecord(start);
    transposeNaive<<<dimGrid, dimBlock>>>(d_out, d_in, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);
    CHECK(cudaMemcpy(h_out_naive, d_out, MEM_SIZE, cudaMemcpyDeviceToHost));
    printf("Naive Kernel Time:    %.3f ms\n", ms_naive);

    // --------------------------------------------------------
    // 测试 2: Optimized Kernel
    // --------------------------------------------------------
    CHECK(cudaMemset(d_out, 0, MEM_SIZE));
    cudaEventRecord(start);
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_out, d_in, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_opt, start, stop);
    CHECK(cudaMemcpy(h_out_opt, d_out, MEM_SIZE, cudaMemcpyDeviceToHost));
    printf("Optimized Kernel Time: %.3f ms\n", ms_opt);

    // --------------------------------------------------------
    // 结果验证与分析
    // --------------------------------------------------------
    if (ms_opt > 0)
        printf("Speedup:              %.2fx\n", ms_naive / ms_opt);
    
    printf("\nVerifying results...\n");
    bool passed = true;
    for (int i = 0; i < N * N; i++) {
        // 验证逻辑：Naive 的结果应该是正确的基准
        if (abs(h_out_naive[i] - h_out_opt[i]) > 1e-5) {
            printf("Mismatch at index %d: Naive=%f, Opt=%f\n", i, h_out_naive[i], h_out_opt[i]);
            passed = false;
            break;
        }
        // 也稍微验证一下是否真的转置了 (检查 h_in)
        int row = i / N; int col = i % N;
        if (abs(h_out_opt[i] - h_in[col * N + row]) > 1e-5) {
             printf("Transpose logic error at %d\n", i);
             passed = false;
             break;
        }
    }

    if (passed) printf("TEST PASSED! Great job!\n");
    else printf("TEST FAILED.\n");

    // 清理资源
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out_naive); free(h_out_opt);
    return 0;
}