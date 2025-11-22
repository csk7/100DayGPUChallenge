#include<iostream>
#include<cuda.h>
#include <cassert>

#define TX_PER_BLOCK 512
#define BK 8
#define BN 64
#define BM 64
#define TM (BM)/(TX_PER_BLOCK/BN)
#define CUDA_CHECK(call) \
    cudaError_t err = call;\
    if(err != cudaSuccess) \
        printf("Error %s at %d:",cudaGetErrorString(err),__LINE__); 
                        

#define CEIL_CUSTOM(M, N) (((M) + (N) - 1)/(N))

float** assignHostSpace(int rows, int cols)
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

void assignHostValues(float** hostArr, int rows, int cols)
{
    for(int i=0;i<rows;i++)
    {
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = float(uint(i*cols + j)%100);
        }
    }
}

void matMulCpu(float** h_A, float** h_B, float** h_C, int M, int K, int N)
{
    float pSum = 0.0;
    for(int i = 0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            pSum = 0.0;
            for(int inLoop = 0; inLoop<K; inLoop++)
            {
                pSum+= (h_A[i][inLoop]*h_B[inLoop][j]);
            }
            h_C[i][j] = pSum;
        }
    }
}

void mismatch2D(float** cpuArr, float** gpuArr, int row, int col)
{
    int flag = 1;
    for(int i=0; i<row; i++)
    {
        for(int j = 0; j<col; j++)
        {
            if(cpuArr[i][j] != gpuArr[i][j])
            {
                flag = 0;
                printf("Mismatch at : (%d,%d), CPU: %f ; GPU: %f \n",i,j,cpuArr[i][j],gpuArr[i][j]);
            }
        }
    }
    if(flag == 1)
        printf("Success \n");
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

void convert_1D_to_2D(float* inpArr, float** outArr, int rows, int cols)
{
    for(int i=0 ;i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            outArr[i][j] = inpArr[i*cols + j];
        }
    }
}

__global__ void matMulKernel(float* d_A, float* d_B, float* d_C, int M, int K, int N)
{

    d_A += blockIdx.y*BM*K;
    d_B += blockIdx.x*BN;
    d_C += blockIdx.y*BM*N + blockIdx.x*BN;

    const int innerRowA = threadIdx.x/BK;
    const int innerColA = threadIdx.x%BK;

    const int innerRowB = threadIdx.x/BN;
    const int innerColB = threadIdx.x%BN;

    const int idxRow = threadIdx.x/BN;
    const int idxCol = threadIdx.x%BN;

    extern __shared__ float shMem[];
    
    float* MdS = shMem;
    float* NdS = &shMem[BM*BK];

    float pSum[TM] = {0.0};    

    for(int ph = 0; ph<K; ph+=BK)
    {
        //Load shared Mem
        MdS[innerRowA*BK + innerColA] = d_A[innerRowA*K + innerColA];
        NdS[innerRowB*BN + innerColB] = d_B[innerRowB*N + innerColB];
        __syncthreads();

        d_A += BK;
        d_B += (BK*N);
        //Partial dot product
        for(int idxK=0; idxK<BK; idxK++)
        {
            float tmp = NdS[idxK*BN + idxCol];
            for(int idxM=0; idxM<TM; idxM++)
            {
                pSum[idxM] += (MdS[(idxRow*TM+idxM)*BK + idxK]*tmp);
            }
        }
        __syncthreads();
    }
    
    for(int idxM=0; idxM<TM; idxM++)
    {
        if((idxRow*TM + idxM)<M and idxCol<N)
            d_C[(idxRow*TM + idxM)*N + idxCol] = pSum[idxM];
    }
    
    
}

void matMulGpu(float** h_A, float** h_B, float** h_C, int M, int K, int N)
{

    //Prep Host values
    float* h_A_1D = convert_2D_to_1D(h_A, M, K);
    float* h_B_1D = convert_2D_to_1D(h_B, K, N);

    //Declare device variables
    float* d_A;
    float* d_B;
    float* d_C;
    int sizeA = M * K * sizeof(float);
    int sizeB = K * N * sizeof(float);
    int sizeC = M * N * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    //Copy values to device
    cudaMemcpy(d_A, h_A_1D, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_1D, sizeB, cudaMemcpyHostToDevice);

    //Call Kernel
    dim3 blockSize(TX_PER_BLOCK,1,1);
    assert(TX_PER_BLOCK == (BM*BK));
    assert(TX_PER_BLOCK == (BK*BN));
    dim3 gridSize(CEIL_CUSTOM(N,BN),CEIL_CUSTOM(M,BM), 1);
    size_t shMemSize = (BM*BK + BK*BN)*sizeof(float);

    matMulKernel<<<gridSize, blockSize, shMemSize>>>(d_A, d_B, d_C, M, K, N);
    //Copy values back
    float* h_C_1D = new float[M*N];
    cudaMemcpy(h_C_1D, d_C, sizeC, cudaMemcpyDeviceToHost);
    convert_1D_to_2D(h_C_1D, h_C, M, N);
    
    //Free device space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    //Declare host variables
    int M = 8162;
    int K = 6144;
    int N = 4092;

    float** h_A = assignHostSpace(M, K);
    float** h_B = assignHostSpace(K, N);
    float** h_C_cpu = assignHostSpace(M, N);
    float** h_C_gpu = assignHostSpace(M, N);

    //Assign values
    assignHostValues(h_A, M, K);
    assignHostValues(h_B, K, N);

    //Call CPU and GPU
    //matMulCpu(h_A, h_B, h_C_cpu, M, K, N);
    matMulGpu(h_A, h_B, h_C_gpu, M, K, N);

    //compare
    //mismatch2D(h_C_cpu, h_C_gpu, M, N);

    return 0;
}
