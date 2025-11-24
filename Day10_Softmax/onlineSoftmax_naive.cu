#include<iostream>
#include<cuda.h>
#include<cmath>
#include<random>

using namespace std;

#define PRINT_FLAG false
#define TX_PER_BLOCK 256

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
    mt19937 gen(2025);  // fixed seed for determinism
    uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
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



__global__ void softmaxKernel(float* d_A, float* d_C, int M, int N)
{
    const int idxM = blockIdx.x*blockDim.x + threadIdx.x;
    if(idxM > M) return;

    d_A += idxM * N; 
    d_C += idxM * N;

    float globalMax = -INFINITY;
    float sum = 0.0;
    float norm = 0.0; 

    for(int idxN = 0; idxN<N; idxN++)
    {
        float val = d_A[idxN];
        if(val > globalMax)
        {
            norm *= expf(globalMax - val);
            globalMax = val;
        }
        sum = norm + expf(val - globalMax);
        norm = sum;
    }
    for(int idxN = 0; idxN<N; idxN++)
    {
        float val = d_A[idxN];
        d_C[idxN] = expf(val - globalMax)/sum;
    }

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

    dim3 gridSize(CEIL_CUSTOM(M,TX_PER_BLOCK),1, 1);
    //size_t shMemSize = (2*BM*BK + 2*BK*BN)*sizeof(float) + 2 * sizeof(cuda::barrier<cuda::thread_scope::thread_scope_block>);

    softmaxKernel<<<gridSize, blockSize>>>(d_A, d_C, M, N);
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

    if(printFlag)
    {
        h_A[0][0] = {3.0};
        h_A[0][1] = {2.0};
        h_A[0][2] = {5.0};
        h_A[0][3] = {1.0};
    }
    else
    {
        //Assign values
        assignHostValues(h_A, M, N);
    }

    //Call CPU and GPU
    if(printFlag)
    {
        softmaxCpu(h_A, h_C_cpu, M, N);
    }
    softmaxGpu(h_A, h_C_gpu, M, N);

    //compare
    if(printFlag)
    {
        mismatch2D(h_C_cpu, h_C_gpu, M, N);
    }    

    return 0;
}