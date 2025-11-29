#include<cuda.h>
#include<iostream>

#include "gemv_utils.cuh"
#define TX_PER_BLOCK 256 //(BM*BN)/(TM) == TX_PER_BLOCK
#define WARPSIZE 32 

#define BN 64
#define BM 16

#define TM 4 //N_TILES in Y dir

#define CEIL_CUSTOM(M, N) (((M) + (N) - 1)/(N))
#define CUDA_CHECK(call)               \
    do{                                \
        cudaError_t err = call;        \
        if(err != cudaSuccess)         \
            printf("Error %s at line %d",cudaGetErrorString(err), __LINE__);  \
    }while(0) 

float** assignHostSpace2D(int rows, int cols)
{
    float** hostArr;
    hostArr = new float*[rows];
    for(int i=0; i<rows; i++)
    {
        hostArr[i] = new float[cols];
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = 0;
        }
    }
    return hostArr;
}

void assignHostValues2D(float** hostArr, int rows, int cols)
{
    for(int i=0;i<rows;i++)
    {
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = float(uint(i*cols + j)%100);
        }
    }
}

float* assignHostSpace1D(int rows)
{
    float* hostArr;
    hostArr = new float[rows];
    for(int i=0; i<rows; i++)
    {
        hostArr[i] = 0;
    }
    return hostArr;
}

void assignHostValues1D(float* hostArr, int rows)
{
    for(int i=0;i<rows;i++)
    {
        hostArr[i] = float(i%100);
    }
}

void gemvCpu(float** h_A, float* h_B, float* h_C, int M, int N)
{
    float pSum = 0.0;
    for(int i = 0; i<M; i++)
    {
        pSum = 0.0;
        for(int j=0; j<N; j++)
        {
            pSum += (h_A[i][j]*h_B[j]);
        }
        h_C[i] = pSum;
    }
}

float* convert_2D_to_1D(float** inpArr, int rows, int cols)
{
    float* outArr;
    outArr = new float[rows*cols];
    for(int i = 0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            outArr[i*cols + j] = inpArr[i][j];
        }
    }
    return outArr;
}

void mismatch1D(float* cpuArr, float* gpuArr, int row)
{
    int flag = 1;
    for(int i=0; i<row; i++)
    {
        if(cpuArr[i] != gpuArr[i])
        {
            flag = 0;
            printf("Mismatch at : (%d), CPU: %f ; GPU: %f \n",i,cpuArr[i],gpuArr[i]);
        }
    }
    if(flag == 1)
        printf("Success \n");
}


__global__ void __launch_bounds__((BM*BN)/(TM),1) gemvKernel(float* d_A, float* d_B, float* d_C, int M, int N)
{
    __shared__ float shMem[BM/TM*BN/WARPSIZE];
    if(blockIdx.x*BM > M) return;

    d_A += blockIdx.x*BM*N;
    d_C += blockIdx.x*BM;
    
    const int idxRow = threadIdx.x / BN;
    const int idxCol = threadIdx.x % BN;

    float pVal[TM] = {0.0};

    for(int idxN = idxCol; idxN<N; idxN+=BN)
    {
        float regB = d_B[idxCol];
        for(int idxTile=0; idxTile<TM; idxTile++)
        {
            pVal[idxTile] += d_A[idxRow*N + idxN] * regB;
            d_A += (BM/TM)*N;
        }
        d_A -= BM*N;
        d_B += BN;
    }

    __syncthreads();
    for(int idxTile=0; idxTile<TM; idxTile++)
    {
        //printf("Thread (%d) Before Val: %f \n", threadIdx.x, pVal[idxTile]);
        blockSum(pVal[idxTile], &shMem[0], 0.0f, BN);
        if(idxCol == 0)
            pVal[idxTile] = shMem[idxRow*BN/WARPSIZE];
        //printf("Thread (%d) Final Val: %f \n", threadIdx.x, pVal[idxTile]);
    }
    
    if(idxCol == 0)
    {
        for(int idxTile=0; idxTile<TM; idxTile++)
        {
            d_C[idxRow] = pVal[idxTile];
            d_C += (BM/TM);
        }
    }
}

void gemvGpu(float** h_A, float* h_B, float* h_C, int M, int N)
{

    //Prep Host values
    float* h_A_1D = convert_2D_to_1D(h_A, M, N);

    //Declare device variables
    float* d_A;
    float* d_B;
    float* d_C;
    int sizeA = M * N * sizeof(float);
    int sizeB = N * sizeof(float);
    int sizeC = M * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    //Copy values to device
    cudaMemcpy(d_A, h_A_1D, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    //Call Kernel
    dim3 blockSize(TX_PER_BLOCK,1,1);

    dim3 gridSize(CEIL_CUSTOM(M,BM), 1, 1);

    gemvKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N);
    //Copy values back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    //Free device space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{

    //Declare host variables
    int M = 4;
    int N = 128;

    float** h_A = assignHostSpace2D(M, N);
    float* h_B = assignHostSpace1D(N);
    float* h_C_cpu = assignHostSpace1D(M);
    float* h_C_gpu = assignHostSpace1D(M);

    //Assign values
    assignHostValues2D(h_A, M, N);
    assignHostValues1D(h_B, N);

    //Call CPU and GPU
    gemvCpu(h_A, h_B, h_C_cpu, M, N);
    gemvGpu(h_A, h_B, h_C_gpu, M, N);

    //compare
    mismatch1D(h_C_cpu, h_C_gpu, M);

    return 0;
    
}