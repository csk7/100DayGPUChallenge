#include<iostream>
#include<cuda.h>

#define TX_PER_BLOCK 1024
#define PRINT_FLAG true

#define CEIL_CUSTOM(M,N) ((M) + (N) - 1)/(N)
#define CUDA_CHECK(call) \
    do{ \
        cudaError_t err = call; \
        if(err != cudaSuccess)  \
            printf("Error : %s at line : %d \n",cudaGetErrorString(err),__LINE__); \
    }while(0)


float* assignHostSpace(int size)
{
    float* hostArr;
    hostArr = new float[size];
    for(int i=0; i<size; i++)
    {
        hostArr[i] = 0;
    }
    return hostArr;
}

void assignHostValues(float* hostArr, int size)
{
    for(int i=0; i<size; i++)
    {
        hostArr[i] = i%1000;
    }
}

void reductionCpu(float* h_A, float* h_C, int size)
{
    float pSum = 0.0;
    for(int i = 0; i<size; i++)
    {
        pSum+= h_A[i];
    }
    h_C[0] = pSum;
    
}

void mismatch1D(float* cpuArr, float* gpuArr, int size)
{
    int flag = 1;
    for(int i=0; i<size; i++)
    {
        if(cpuArr[i] != gpuArr[i])
        {
            flag = 0;
            printf("Mismatch at : (%d), CPU: %f ; GPU: %f \n",i, cpuArr[i], gpuArr[i]);
        }
    }
    if(flag == 1)
        printf("Success \n");
}

__global__ void reduction(float* d_A, float* d_C, int size)
{
    extern __shared__ float shMem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shMem[threadIdx.x] = d_A[idx];
    __syncthreads();
    
    for(int i = (blockDim.x/2); i>=1; i/=2)
    {
        float tmpReg = 0.0;
        if(threadIdx.x < i)
        {
            tmpReg = shMem[threadIdx.x + i];
        }
        __syncthreads();
        shMem[threadIdx.x] += tmp;
        __syncthreads();
    }
    d_C[0] = shMem[0];
}

void reductionGpu(float* h_A, float* h_C, int size)
{

    //Declare device variables
    float* d_A;
    float* d_C;
    int sizeA = size * sizeof(float);
    int sizeC = 1 * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    //Copy values to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);

    //Call Kernel
    dim3 blockSize(TX_PER_BLOCK,1,1);

    dim3 gridSize(CEIL_CUSTOM(size,TX_PER_BLOCK), 1, 1);
    size_t shMemSize = (TX_PER_BLOCK)*sizeof(float);

    reduction<<<gridSize, blockSize, shMemSize>>>(d_A, d_C, size);
    //Copy values back

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    //Free device space
    cudaFree(d_A);
    cudaFree(d_C);
}

int main()
{
    //Declare host variables
    bool printFlag = PRINT_FLAG;
    int size;
    if(printFlag)
    {
        size = 256;
    }
    else
    {
        size = 8162;
    }

    float* h_A = assignHostSpace(size);
    float* h_C_cpu = assignHostSpace(1);
    float* h_C_gpu = assignHostSpace(1);

    //Assign values
    assignHostValues(h_A, size);

    //Call CPU and GPU
    if(printFlag)
    {
        reductionCpu(h_A, h_C_cpu, size);
    }
    reductionGpu(h_A, h_C_gpu, size);

    //compare
    if(printFlag)
    {
        mismatch1D(h_C_cpu, h_C_gpu, 1);
    }

    return 0;
}