// gray_scale
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 图像大小：假设我们有 100 万个像素
#define N 1000000

// =========================================================
// TODO 1: 编写 Kernel 函数
// 要求：
// 1. 计算全局索引 i
// 2. 进行边界检查 (防止越界)
// 3. 实现灰度公式: Gray = 0.21*r + 0.72*g + 0.07*b
// =========================================================
__global__ void colorToGrayscaleConversion(float* Pout, float* Pin_r, float* Pin_g, float* Pin_b, int n) {
    // 1. 计算全局线程索引
    // int i = ...
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // 2. 边界检查与计算
    // if (...) {
    //     Pout[i] = ...
    // }
    if (i < n) {
        Pout[i] = 0.21 * Pin_r[i] + 0.72 * Pin_g[i] + 0.07 * Pin_b[i];
    }
}

int main() {
    // 1. 申请 Host 内存
    float *h_r, *h_g, *h_b, *h_gray;
    size_t size = N * sizeof(float);
    
    h_r = (float*)malloc(size);
    h_g = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_gray = (float*)malloc(size);

    // 初始化数据 (随机生成 0-1 之间的颜色值)
    for (int i = 0; i < N; ++i) {
        h_r[i] = rand() / (float)RAND_MAX;
        h_g[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // =========================================================
    // TODO 2: 申请 Device (GPU) 内存
    // 提示：需要申请 d_r, d_g, d_b, d_gray 四块显存
    // 使用 cudaMalloc
    // =========================================================
    float *d_r, *d_g, *d_b, *d_gray;
    // cudaMalloc...
    // cudaMalloc...
    // cudaMalloc...
    // cudaMalloc...
    cudaMalloc ((void**)&d_r, size);
    cudaMalloc ((void**)&d_g, size);
    cudaMalloc ((void**)&d_b, size);
    cudaMalloc ((void**)&d_gray, size);


    // =========================================================
    // TODO 3: 将数据从 Host 拷贝到 Device
    // 提示：使用 cudaMemcpy，方向是 cudaMemcpyHostToDevice
    // 注意：只需要拷贝 r, g, b 三个输入数组，gray 不需要拷贝过去
    // =========================================================
    // cudaMemcpy...
    // cudaMemcpy...
    // cudaMemcpy...
    cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);


    // =========================================================
    // TODO 4: 配置 Kernel 启动参数
    // 提示：
    // 1. 定义 block 大小为 256
    // 2. 计算 grid 大小，确保能覆盖 N 个元素 (使用 ceil 逻辑)
    // =========================================================
    int threadsPerBlock = 256;
    int blocksPerGrid = ceil(N/(float)threadsPerBlock); // 修改这里


    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // =========================================================
    // TODO 5: 启动 Kernel
    // =========================================================
    // colorToGrayscaleConversion<<<...>>>(...);
    colorToGrayscaleConversion<<<blocksPerGrid, threadsPerBlock>>>(d_gray, d_r, d_g, d_b, N);


    // 检查 Kernel 是否出错 (可选，但推荐)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // =========================================================
    // TODO 6: 将结果从 Device 拷贝回 Host
    // 提示：使用 cudaMemcpy，方向是 cudaMemcpyDeviceToHost
    // =========================================================
    // cudaMemcpy...
    cudaMemcpy(h_gray, d_gray, size, cudaMemcpyDeviceToHost);


    // 验证结果 (CPU 算一遍对比)
    printf("Verifying results...\n");
    int errorCount = 0;
    for (int i = 0; i < N; ++i) {
        float expected = 0.21f * h_r[i] + 0.72f * h_g[i] + 0.07f * h_b[i];
        if (fabs(expected - h_gray[i]) > 1e-5) {
            errorCount++;
            if (errorCount < 5) {
                printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, expected, h_gray[i]);
            }
        }
    }

    if (errorCount == 0) {
        printf("PASSED! All results match.\n");
    } else {
        printf("FAILED! Total errors: %d\n", errorCount);
    }

    // =========================================================
    // TODO 7: 释放 Device 内存
    // 提示：使用 cudaFree
    // =========================================================
    // cudaFree...
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_gray);


    // 释放 Host 内存
    free(h_r); free(h_g); free(h_b); free(h_gray);

    return 0;
}