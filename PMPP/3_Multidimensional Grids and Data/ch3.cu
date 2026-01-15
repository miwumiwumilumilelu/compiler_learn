#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// =============================================================
// 配置参数
// =============================================================
#define MATRIX_SIZE 32      // 矩阵大小 32x32 (用于 Task 1)
#define IMG_WIDTH   64      // 图像宽度 (用于 Task 2)
#define IMG_HEIGHT  64      // 图像高度 (用于 Task 2)
#define BLUR_SIZE   1       // 模糊半径 (Patch大小为 3x3)

// 错误检查宏
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

// =============================================================
// TASK 1: 矩阵乘法 Kernel (对应 3.4 节)
// 计算 C = A * B
// A, B, C 都是 Width x Width 的方阵
// =============================================================
__global__ void matrixMulKernel(float* A, float* B, float* C, int Width) {
    // ---------------------------------------------------------
    // TODO 1: 计算当前线程负责的 row (行) 和 col (列)
    // 提示：使用 2D 的 blockIdx, blockDim, threadIdx
    // ---------------------------------------------------------
    int row = blockDim.x * blockIdx.x + threadIdx.x; // 修改这里
    int col = blockDim.y * blockIdx.y + threadIdx.y; // 修改这里

    if (row < Width && col < Width) {
        float sum = 0.0f;
        // -----------------------------------------------------
        // TODO 2: 实现矩阵乘法的点积循环
        // 提示：遍历 k 从 0 到 Width
        // 注意：A 是行主序 (row * Width + k)
        //       B 是行主序 (k * Width + col)
        // -----------------------------------------------------
        // for (...) {
        //     sum += ...
        // }
        for (int k = 0; k < Width; k++) {
            sum += A[row * Width + k] * B[k * Width + col];
        }
        
        C[row * Width + col] = sum;
    }
}

// =============================================================
// TASK 2: 图像模糊 Kernel (对应 3.3 节)
// 使用 Box Blur (均值模糊)
// =============================================================
__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    // ---------------------------------------------------------
    // TODO 3: 计算 2D 坐标 (col, row) 并进行基本的越界检查
    // ---------------------------------------------------------
    int col = blockDim.x * blockIdx.x + threadIdx.x; // 修改这里
    int row = blockDim.y * blockIdx.y + threadIdx.y; // 修改这里

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        // -----------------------------------------------------
        // TODO 4: 编写嵌套循环遍历 Patch (邻域)
        // 提示：blurRow 从 -BLUR_SIZE 到 +BLUR_SIZE
        //       blurCol 从 -BLUR_SIZE 到 +BLUR_SIZE
        // -----------------------------------------------------
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // ---------------------------------------------
                // TODO 5: 邻居的边界检查 (关键！)
                // 只有当 curRow 和 curCol 都在图像范围内时，才累加
                // ---------------------------------------------
                // if (...) {
                //      pixVal += in[...];  // 记得把 2D 坐标转 1D 索引
                //      pixels++;
                // }
                if (curRow < h && curRow >= 0 && curCol < w && curCol >=0) {
                    pixVal += in[curRow * w + curCol];
                    pixels ++;
                }
            }
        }

        // 写回结果 (取平均)
        if (pixels > 0) {
            out[row * w + col] = (unsigned char)(pixVal / pixels);
        }
    }
}

// =============================================================
// 主函数
// =============================================================
int main() {
    printf("Starting CUDA Final Review Assignment...\n");

    // ------------------- 准备 Task 1 数据 -------------------
    int m_bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(m_bytes);
    h_B = (float*)malloc(m_bytes);
    h_C = (float*)malloc(m_bytes);

    // 初始化 A 和 B
    for(int i=0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        h_A[i] = 1.0f; h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, m_bytes));
    CHECK(cudaMalloc((void**)&d_B, m_bytes));
    CHECK(cudaMalloc((void**)&d_C, m_bytes));

    CHECK(cudaMemcpy(d_A, h_A, m_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, m_bytes, cudaMemcpyHostToDevice));

    // ---------------------------------------------------------
    // TODO 6: 定义 Task 1 的执行配置 (dim3)
    // 要求：Block 大小为 16x16
    // Grid 大小要足以覆盖 MATRIX_SIZE (使用 ceil 逻辑)
    // ---------------------------------------------------------
    dim3 dimBlockMat(16, 16, 1); // 修改这里
    dim3 dimGridMat(ceil(IMG_WIDTH/16.0), ceil(IMG_HEIGHT/16.0));  // 修改这里

    printf("Launching Matrix Mul Kernel...\n");
    matrixMulKernel<<<dimGridMat, dimBlockMat>>>(d_A, d_B, d_C, MATRIX_SIZE);
    CHECK(cudaDeviceSynchronize());
    
    // 验证 Task 1
    CHECK(cudaMemcpy(h_C, d_C, m_bytes, cudaMemcpyDeviceToHost));
    if (h_C[0] == 2.0f * MATRIX_SIZE) printf(">> Task 1 (MatMul): PASSED!\n");
    else printf(">> Task 1 (MatMul): FAILED! Expected %.1f, got %.1f\n", 2.0f * MATRIX_SIZE, h_C[0]);


    // ------------------- 准备 Task 2 数据 -------------------
    int img_bytes = IMG_WIDTH * IMG_HEIGHT * sizeof(unsigned char);
    unsigned char *h_in, *h_out;
    h_in = (unsigned char*)malloc(img_bytes);
    h_out = (unsigned char*)malloc(img_bytes);

    // 初始化一张只有中间有个白点的黑图
    for(int i=0; i<IMG_WIDTH*IMG_HEIGHT; i++) h_in[i] = 0;
    h_in[(IMG_HEIGHT/2) * IMG_WIDTH + (IMG_WIDTH/2)] = 255; // 中心点最亮

    unsigned char *d_in, *d_out;
    CHECK(cudaMalloc((void**)&d_in, img_bytes));
    CHECK(cudaMalloc((void**)&d_out, img_bytes));

    CHECK(cudaMemcpy(d_in, h_in, img_bytes, cudaMemcpyHostToDevice));

    // ---------------------------------------------------------
    // TODO 7: 定义 Task 2 的执行配置 (dim3)
    // 要求：Block 大小为 16x16
    // Grid 大小要足以覆盖 IMG_WIDTH 和 IMG_HEIGHT
    // ---------------------------------------------------------
    dim3 dimBlockImg(16, 16, 1); // 修改这里
    dim3 dimGridImg((IMG_WIDTH-1)/16 + 1, (IMG_HEIGHT-1)/16 + 1);  // 修改这里

    printf("Launching Blur Kernel...\n");
    blurKernel<<<dimGridImg, dimBlockImg>>>(d_in, d_out, IMG_WIDTH, IMG_HEIGHT);
    CHECK(cudaDeviceSynchronize());

    // 验证 Task 2 (检查中心点周围是否被模糊了)
    CHECK(cudaMemcpy(h_out, d_out, img_bytes, cudaMemcpyDeviceToHost));
    int centerIdx = (IMG_HEIGHT/2) * IMG_WIDTH + (IMG_WIDTH/2);
    // 原始是255，周围8个是0。平均值应该是 255 / 9 ≈ 28
    if (h_out[centerIdx] > 0 && h_out[centerIdx] < 255) printf(">> Task 2 (Blur): PASSED! Center pixel value: %d\n", h_out[centerIdx]);
    else printf(">> Task 2 (Blur): FAILED! Center pixel is %d (Expected approx 28)\n", h_out[centerIdx]);

    // 释放内存
    free(h_A); free(h_B); free(h_C);
    free(h_in); free(h_out);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}