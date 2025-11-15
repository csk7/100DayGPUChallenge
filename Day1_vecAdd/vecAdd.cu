#include<iostream>
#include<cuda.h>

void cpuVecAdd(float* h_A, float* h_B, float* h_C, int N)
{
    for(int i=0; i<N; i++)
    {
        h_C[i] = h_A[i] + h_B[i];
    }
}

void mismatch(float* cpuArr, float* gpuArr, int N)
{
    int flag = 1;
    for(int i=0; i<N; i++)
    {
        if(cpuArr[i] != gpuArr[i])
        {
            flag = 0;
            printf("Mismatch at : %d, CPU: %f ; GPU: %f \n",i,cpuArr[i],gpuArr[i]);
        }
    }
    if(flag == 1)
        printf("Success \n");
}

__global__ void vecAddKernel(float* d_A, float* d_B, float* d_C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

void gpuVecAdd(float* h_A, float* h_B, float* h_C, int N)
{
    //Declare device variables
    float* d_A;
    float* d_B;
    float* d_C;

    int size = N* sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    //Transfer data
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //Call kernel
    dim3 blockSize(256,1,1);
    dim3 gridSize(((N + blockSize.x - 1)/blockSize.x),1,1);

    vecAddKernel<<<gridSize,blockSize>>>(d_A, d_B, d_C, N);
    //Transfer results
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    //Free data
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
int main()
{
    int N = 5;

    float* h_A;
    float* h_B;
    float* h_C_cpu;
    float* h_C_gpu;

    h_A = new float[N];
    h_B = new float[N];
    h_C_cpu = new float[N];
    h_C_gpu = new float[N];

    for(int i=0; i<N; i++)
    {
        h_A[i] = i;
        h_B[i] = 2*i;
    }

    cpuVecAdd(h_A, h_B, h_C_cpu, N);
    gpuVecAdd(h_A, h_B, h_C_gpu, N);

    mismatch(h_C_cpu, h_C_gpu, N);

    return 0;

}