#include<iostream>
#include<cuda.h>
#include<cmath>
#include<random>
#include<cassert>

using namespace std;

#define PRINT_FLAG false
#define TX_PER_BLOCK 1024

#define CEIL_CUSTOM(M,N) ((M) + (N) - 1)/(N)
#define CUDA_CHECK(call) \
    do{ \
        cudaError_t err = call; \
        if(err != cudaSuccess)  \
            printf("Error : %s at line : %d \n",cudaGetErrorString(err),__LINE__); \
    }while(0)

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
    mt19937 gen(2026);  // fixed seed for determinism
    uniform_real_distribution<float> dist(-0.5f, 0.5f);
    
    for(int i=0;i<rows;i++)
    {
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = dist(gen);
        }
    }
}


void mismatch2D(float** cpuArr, float** gpuArr, int row, int col)
{
    int flag = 1;
    const float epsilon = 1e-6f; // 5 decimal places tolerance
    for(int i=0; i<row; i++)
    {
        for(int j = 0; j<col; j++)
        {
            if(fabsf(cpuArr[i][j] - gpuArr[i][j]) > epsilon)
            {
                flag = 0;
                printf("Mismatch at : (%d,%d), CPU: %f ; GPU: %f \n",i,j,cpuArr[i][j],gpuArr[i][j]);
            }
        }
    }
    if(flag == 1)
        printf("Success \n");
}

void softmaxCpu(float** h_A, float** h_C, int M, int N)
{
    float sum = 0.0;
    float globalMax = -INFINITY;
    float val;

    for(int i = 0; i<M; i++)
    {
        sum = 0.0;
        globalMax = -INFINITY;

        //Pass 1
        for(int j=0; j<N; j++)
        {
            val = h_A[i][j];
            if(val > globalMax)
            {
                globalMax = val;
            }
        }

        //Pass 2
        for(int j=0; j<N; j++)
        {
            val = h_A[i][j];
            sum += expf(val - globalMax);
        }

        //Pass 3
        for(int j=0; j<N; j++)
        {
            val = h_A[i][j];
            h_C[i][j] = expf(val-globalMax)/sum;
            //printf("%f \t",h_C[i][j]);
        }
        //printf("\n");
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


__global__ void softmaxKernel(float* __restrict__ d_A, float* __restrict__ d_C, int M, int N)
{
    if(blockIdx.x > M) return;
    //Declare shMem to collate results adter each local run
    __shared__ float shMem[TX_PER_BLOCK];

    float localMax = -INFINITY;
    float localNorm = 0.0f;
    //start address
    d_A += blockIdx.x*N;
    d_C += blockIdx.x*N;

    const int tid = threadIdx.x;
    //Collate local results 
    for(int idxN = tid; idxN<N; idxN += blockDim.x)
    {
        float val = d_A[idxN];
        if(val > localMax)
        {
            localNorm *= expf(localMax - val);
            localMax = val;
        }
        localNorm += expf(val - localMax);
    }

    shMem[tid] = localMax;
    __syncthreads();
    //collate shMem results
    for(int level=blockDim.x/2; level>=1; level /= 2)
    {
        int stride = level;
        if(tid<level)
        {
            shMem[tid] = max(shMem[tid], shMem[tid+stride]);
        }
        __syncthreads();
    }

    float gMax = shMem[0];
    shMem[tid] = localNorm*expf(localMax - gMax);
    __syncthreads();

    for(int level=blockDim.x/2; level>=1; level /= 2)
    {
        int stride = level;
        if(tid<level)
        {
            shMem[tid] +=  shMem[tid + stride];
        }
        __syncthreads();
    }
    
    float gSum = shMem[0];
    __syncthreads();

    for(int idxN = tid; idxN<N; idxN += blockDim.x)
    {
        d_C[idxN] = expf(d_A[idxN] - gMax)/gSum;
    }
    /*
        float val = shMem[idxRow*BN + idxCol];
        if(val > globalMax)
        {
            norm *= expf(globalMax - val);
            globalMax = val;
        }
        sum = norm + expf(val - globalMax);
        norm = sum;
    */

}


void softmaxGpu(float** h_A, float** h_C, int M, int N)
{


    //Prep Host values
    float* h_A_1D = convert_2D_to_1D(h_A, M, N);

    //Declare device variables
    float* d_A;
    float* d_C;
    int sizeA = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeA));

    //Copy values to device
    cudaMemcpy(d_A, h_A_1D, sizeA, cudaMemcpyHostToDevice);

    //Call Kernel
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, 0);

    dim3 blockSize(TX_PER_BLOCK,1,1);

    dim3 gridSize(M, 1, 1);
    //ize_t shMemSize = (BM*BK)*sizeof(float);

    softmaxKernel<<<gridSize, blockSize>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();

    //Copy values back
    float* h_C_1D = new float[M*N];
    cudaMemcpy(h_C_1D, d_C, sizeA, cudaMemcpyDeviceToHost);
    convert_1D_to_2D(h_C_1D, h_C, M, N);
    
    //Free device space
    cudaFree(d_A);
    cudaFree(d_C);
}



int main()
{
    //Declare host variables
    bool printFlag = PRINT_FLAG;
    int M, N;
    if(printFlag)
    {
        M = 1;
        N = 4;
    }
    else
    {
        M = 2048;
        N = 32768;
    }

    float** h_A = assignHostSpace(M, N);
    float** h_C_cpu = assignHostSpace(M, N);
    float** h_C_gpu = assignHostSpace(M, N);


    //Assign values
    assignHostValues(h_A, M, N);
    //Call CPU and GPU
    //if(printFlag)
    //{
        softmaxCpu(h_A, h_C_cpu, M, N);
    //}
    softmaxGpu(h_A, h_C_gpu, M, N);

    //compare
    //if(printFlag)
    //{
        mismatch2D(h_C_cpu, h_C_gpu, M, N);
    //}    

    return 0;
}