#include "core.h"

__global__ void kernel(int width, int height, float *input, float *output) {
    int i = blockIdx.x / width;
    int j = blockIdx.x % width;
    int tx = threadIdx.x;

    // 初始化共享变量 计算元素个数
    __shared__ int counts[NUMLEN];
    if(tx < NUMLEN){
        counts[tx] = 0;
    }
    __syncthreads();

    // 利用原子操作计算元素个数
    int m = i - 2 + tx / width;
    int n = j - 2 + tx % width;
    if(0 <= m && m < height && 0 <= n && n < width){
        int num = input[m*width+n];
        atomicAdd(&counts[num], 1);
    }
    __syncthreads();

    // 计算合法区域大小
    int up = i >= 2 ? 2 : i;
    int down = height-1-i >= 2 ? 2 : height-1-i;
    int left = j >= 2 ? 2 : j;
    int right = width-1-j >= 2 ? 2 : width-1-j;
    int sum = (up + 1 + down) * (left + 1 + right);

    // 计算相应log值
    __shared__ float logCounts[NUMLEN];
    if(tx < NUMLEN){
        int count = counts[tx];
        if(count != 0){
            logCounts[tx] = count * logf(count) / sum;
        }else{
            logCounts[tx] = 0;
        }
    }
    __syncthreads();

    // 求和
    int t = tx; 
    int k = NUMLEN / 2; 
    while (k != 0) {
        if(t < k){
            logCounts[t] += logCounts[t + k]; 
        } 
        __syncthreads();    
        k /= 2; 
    }

    // 写入结果
    if (tx == 0){
        output[i*width+j] = logf(sum) - logCounts[0];
    }
}

void cudaCallback(int width, int height, float *sample, float **result) {
    int size = width * height;
    float *input_d, *output_d;
    
    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&input_d, sizeof(float)*size));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float)*size));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float)*size, cudaMemcpyHostToDevice));

    // Invoke the device function
    kernel<<< size, 25 >>>(width, height, input_d, output_d);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    *result = (float *)malloc(sizeof(float)*size);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));

    // Note that you don't have to free sample and *result by yourself
}