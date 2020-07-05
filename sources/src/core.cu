#include "core.h"

__global__ void kernel(int width, int height, float *input, float *output) {
    int i = blockIdx.x / width;
    int j = blockIdx.x % width;

    int sum = 0;
    int counts[NUMLEN] = {0};
    for(int m = i-2; m <= i+2; m++){
        for(int n = j-2; n <= j+2; n++){
            if(0 <= m && m < height && 0 <= n && n < width){
                int num = input[m*width+n];
                counts[num]++;
                sum++;
            }
        }
    }

    double res = logf(sum);
    for(int m = 1; m < NUMLEN; m++){
        int count = counts[m];
        if(count != 0){
            res -= logRes[count] / sum;
        }
    }

    output[i*width+j] = res;
}

void cudaCallback(int width, int height, float *sample, float **result) {
    int size = width * height;
    float *input_d, *output_d;
    
    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&input_d, sizeof(float)*size));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float)*size));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float)*size, cudaMemcpyHostToDevice));

    // Invoke the device function
    kernel<<< size, 1 >>>(width, height, input_d, output_d);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    *result = (float *)malloc(sizeof(float)*size);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));

    // Note that you don't have to free sample and *result by yourself
}