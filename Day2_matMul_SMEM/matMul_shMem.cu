#include<iostream>
#include<cuda.h>

#define TILE_WIDTH 16
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
            hostArr[i][j] = i*cols + j;
        }
    }
}

void matMulCpu(float** h_A, float** h_B, float** h_C, int M, int N, int K)
{
    float pSum = 0.0;
    for(int i = 0; i<M; i++)
    {
        for(int j=0; j<K; j++)
        {
            pSum = 0.0;
            for(int inLoop = 0; inLoop<N; inLoop++)
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

__global__ void matMulKernel(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shMem[];
    
    float* MdS = shMem;
    float* NdS = &shMem[TILE_WIDTH*TILE_WIDTH];

    float pSum = 0.0;

    for(int ph = 0; ph<((N+TILE_WIDTH-1)/TILE_WIDTH); ph++)
    {
        //Load shared Mem
        if(row<M && (ph*TILE_WIDTH + threadIdx.x)<N)
            MdS[threadIdx.y*TILE_WIDTH + threadIdx.x] = d_A[row*N + (ph*TILE_WIDTH + threadIdx.x)];
        else
            MdS[threadIdx.y*TILE_WIDTH + threadIdx.x] = 0.0;
        if((ph*TILE_WIDTH + threadIdx.y)<N && col<K)
            NdS[threadIdx.y*TILE_WIDTH + threadIdx.x] = d_B[(ph*TILE_WIDTH + threadIdx.y)*K + col];
        else
            NdS[threadIdx.y*TILE_WIDTH + threadIdx.x] = 0.0;
        __syncthreads();
        //Partial dot product
        for(int i = 0; i<TILE_WIDTH; i++)
        {
            pSum += (MdS[threadIdx.y*TILE_WIDTH + i] * NdS[i*TILE_WIDTH + threadIdx.x]);
        }
        __syncthreads();
    }
    if(row<M && col<K)
    {
        d_C[row*K + col] = pSum;
    }
}

void matMulGpu(float** h_A, float** h_B, float** h_C, int M, int N, int K)
{

    //Prep Host values
    float* h_A_1D = convert_2D_to_1D(h_A, M, N);
    float* h_B_1D = convert_2D_to_1D(h_B, N, K);

    //Declare device variables
    float* d_A;
    float* d_B;
    float* d_C;
    int sizeA = M * N * sizeof(float);
    int sizeB = N * K * sizeof(float);
    int sizeC = M * K * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    //Copy values to device
    cudaMemcpy(d_A, h_A_1D, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_1D, sizeB, cudaMemcpyHostToDevice);

    //Call Kernel
    dim3 blockSize(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridSize(((K + blockSize.x -1)/blockSize.x),((M + blockSize.y -1)/blockSize.y), 1);
    size_t shMemSize = 2*TILE_WIDTH*TILE_WIDTH*sizeof(float);

    matMulKernel<<<gridSize, blockSize, shMemSize>>>(d_A, d_B, d_C, M, N, K);

    //Copy values back
    float* h_C_1D = new float[M*K];
    cudaMemcpy(h_C_1D, d_C, sizeC, cudaMemcpyDeviceToHost);
    convert_1D_to_2D(h_C_1D, h_C, M, K);
    
    //Free device space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    //Declare host variables
    int M = 8162;
    int N = 6144;
    int K = 4092;

    float** h_A = assignHostSpace(M, N);
    float** h_B = assignHostSpace(N, K);
    float** h_C_cpu = assignHostSpace(M, K);
    float** h_C_gpu = assignHostSpace(M, K);

    //Assign values
    assignHostValues(h_A, M, N);
    assignHostValues(h_B, N, K);

    //Call CPU and GPU
    //matMulCpu(h_A, h_B, h_C_cpu, M, N, K);
    matMulGpu(h_A, h_B, h_C_gpu, M, N, K);

    //compare
    //mismatch2D(h_C_cpu, h_C_gpu, M, K);

    return 0;
}
