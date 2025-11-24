#include<iostream>
#include<cuda.h>
#include<cmath>
#include<random>
#include<cassert>

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


template <const int BM, const int BN>
__global__ void softmaxKernel(float* d_A, float* d_C, int M, int N)
{
    //Declared Shared mem
    __shared__ float shMem[BM*BN];
    __shared__ float pMax[BM*BN];
    __shared__ float pSum[BM*BN];
    __shared__ float pNorm[BM*BN];

    __shared__ float gMax[BM]; 
    __shared__ float gSum[BM]; 
    __shared__ float gNorm[BM];
    if(threadIdx.x < BM)
    {
        gMax[threadIdx.x] = -INFINITY;
        gSum[threadIdx.x] = 0.0;
        gNorm[threadIdx.x] = 0.0;
    }
    __syncthreads();

    //Move the globalMem address across rows
    const int blockRows = blockIdx.x * BM;
    if(blockRows > M) return;

    d_A += blockRows * N; 
    d_C += blockRows * N;

    //BN*BM Threads
    const int idxRow = threadIdx.x / BN; //0 to (BM - 1)
    const int idxCol = threadIdx.x % BN; //0 to BN-1


    for(int idxN = 0; idxN<N; idxN+=BN)
    {
        //Load shMem Populate pSum, pNorm, pGmax
        float val = d_A[idxRow*N + idxCol];
        shMem[idxRow*BN + idxCol] = val;
        pMax[idxRow*BN + idxCol] = val;
        pNorm[idxRow*BN + idxCol] = 1.0;
        pSum[idxRow*BN + idxCol] = 1.0;

        __syncthreads();
        //Now do Reduction
        for(int idxLevel = BN/2; idxLevel>=1; idxLevel/=2)
        {
            int stride = idxLevel;
            if(idxCol < idxLevel)
            {
                if(pMax[idxRow*BN + idxCol] < pMax[idxRow*BN + idxCol+stride])
                {
                    pNorm[idxRow*BN + idxCol] *= expf(pMax[idxRow*BN + idxCol] - pMax[idxRow*BN + idxCol+stride]);
                    pMax[idxRow*BN + idxCol] = pMax[idxRow*BN + idxCol+stride];
                }
                else
                {
                    pNorm[idxRow*BN + idxCol+stride] *= expf(pMax[idxRow*BN + idxCol+stride] - pMax[idxRow*BN + idxCol]);
                }
                pSum[idxRow*BN + idxCol] = pNorm[idxRow*BN + idxCol] + pNorm[idxRow*BN + idxCol+stride];
                pNorm[idxRow*BN + idxCol] = pSum[idxRow*BN + idxCol];
            }
        }
        __syncthreads();
        //Collate results to global
        if(idxCol == 0)
        {
            if(pMax[idxRow*BN] > gMax[idxRow])
            {
                gNorm[idxRow] = gNorm[idxRow]*expf(gMax[idxRow]-pMax[idxRow*BN]);
                gMax[idxRow] = pMax[idxRow*BN];
            }
            else
            {
                pNorm[idxRow*BN] = pNorm[idxRow*BN]*expf(pMax[idxRow*BN]-gMax[idxRow]);
            }
            gSum[idxRow] = gNorm[idxRow] + pNorm[idxRow*BN];
            gNorm[idxRow] = gSum[idxRow];
        }
        __syncthreads();
        //House keeping
        d_A += BN;
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
    d_A -= N; 
    for(int idxN = 0; idxN<N; idxN+=BN)
    {
        float val = d_A[idxRow*N + idxCol];
        d_C[idxRow*N + idxCol] = expf(val - gMax[idxRow])/gSum[idxRow];
        d_C += BN;
        d_A += BN;
    }

}


void softmaxGpu(float** h_A, float** h_C, int M, int N)
{

    const int BM = 4;
    const int BN = 64;
    assert(BM*BN == TX_PER_BLOCK);
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

    dim3 gridSize(CEIL_CUSTOM(M,BM),1, 1);
    //ize_t shMemSize = (BM*BK)*sizeof(float);

    softmaxKernel<BM, BN><<<gridSize, blockSize>>>(d_A, d_C, M, N);
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
        M = 4;
        N = 8;
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