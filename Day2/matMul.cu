#include<iostream>
#include<cuda.h>


void assignHostSpace(float** hostArr, int rows, int cols)
{
    hostArr = new float*[rows];
    for(int i=0; i<rows; i++)
    {
        hostArr[i] = new float[cols];
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = 0;
        }
    }
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
    for(int i = 0; i<N; i++)
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

void mismatch2D(float* cpuArr, float* gpuArr, int N)
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

float** convert_1D_to_2D(float* inpArr, int rows, int cols)
{
    float** outArr;
    assignHostSpace(outArr, rows, cols);
    for(int i=0 ;i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            outArr[i][j] = inpArr[i*cols + j];
        }
    }
    return outArr;
}

__global__ void matMulKernel(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float MdS[TILE_WIDTH][TILE_WIDTH];
    __shared__ float NdS[TILE_WIDTH][TILE_WIDTH];

    for(int ph = 0; ph<(N/TILE_WIDTH); ph++)
    {
        
    }
}

void matMulGpu(float** h_A, float** h_B, float** h_C, int M, int N, int K)
{

    //Prep Host values
    float* h_A_1D = convert_2D_to_1D(h_A);
    float* h_B_1D = convert_2D_to_1D(h_B);

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
    cudaMemcpy(d_B, h_B_1D, sizeB, cudaMemcpyHostToDevice;

    //Call Kernel
    dim3 blockSize(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridSize(((K + blockSize.x -1)/blockSize.x),((M + blockSize.y -1)/blockSize.y), 1);

    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    //Copy values back
    cudaMemcpy(h_C_1D, d_C, sizeC, cudaMemcpyDeviceToHost);
    h_C = convert_1D_to_2D(h_C_1D, M, K);
    
    //Free device space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    //Declare host variables
    float** h_A;
    float** h_B;
    float** h_C_cpu;
    float** h_C_gpu;

    int M = 4;
    int N = 4;
    int K = 4;

    assignHostSpace(h_A, M, N);
    assignHostSpace(h_B, N, K);
    assignHostSpace(h_C_cpu, M, K);

    //Assign values
    assignHostValues(h_A, M, N);
    assignHostValues(h_B, N, K);

    //Call CPU and GPU
    matMulCpu(h_A, h_B, h_C_cpu, M, N, K);
    matMulGpu(h_A, h_B, h_C_gpu, M, N, K);

    //compare
    mismatch2D(h_C_cpu, h_C_gpu, M, K);

    return 0;
}
