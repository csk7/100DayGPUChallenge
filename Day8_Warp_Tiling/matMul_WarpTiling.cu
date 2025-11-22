#include<iostream>
#include<cuda.h>
#include <cassert>

#define TX_PER_BLOCK 256
#define WARPSIZE 32

#define BK 16
#define BN 64
#define BM 64
#define TM 4
#define TN 4 //(BM*BN)/(TM*TN) == TX_PER_BLOCK
#define WM 32
#define WN 16
#define WNITERS 1
#define WMITERS 1//(WM*WN)/(WARPSIZE*TM*TN*WNITERS)
#define WSUBM WM/WMITERS
#define WSUBN WN/WNITERS


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

namespace warpKernels
{
    template<const int rowStrideA, const int rowStrideB>
    __device__ void loadFromGlobalToShared(float* d_A, float* d_B, float* MdS, float* NdS, int N, int K, 
                                                int innerRowA, int innerColA, int innerRowB, int innerColB)
    {
        float4 tmp;

        for(int offset = 0; offset<BM; offset+=rowStrideA) //RowstrideA = BM in our cases as data exactly fits.
        {
            tmp = reinterpret_cast<float4*>(&d_A[(innerRowA + offset)*K + 4*innerColA])[0]; 
            MdS[(4*innerColA + 0)*BM + innerRowA + offset] = tmp.x; //We are doing transpose here. It is needed as BK is only 16, No shmem coalescing.
            MdS[(4*innerColA + 1)*BM + innerRowA + offset] = tmp.y; //Swap row and col places, BK --> BN. Also switch while accessing it. Thats it.
            MdS[(4*innerColA + 2)*BM + innerRowA + offset] = tmp.z; // '/' operator tx actually makes subsequent tx access same row/location.
            MdS[(4*innerColA + 3)*BM + innerRowA + offset] = tmp.w;  
        }
        
        for(int offset = 0; offset<BK; offset+=rowStrideB) //RowstrideB = BK in our cases as data exactly fits.
            reinterpret_cast<float4*>(&NdS[(innerRowB+offset)*BN + 4*innerColB])[0] = reinterpret_cast<float4*>(&d_B[(innerRowB+offset)*N + 4*innerColB])[0];
    }

    __device__ void dotProductSharedMem(float* MdS, float* NdS, float* pSum, int warpRow, int warpCol, int subWarpRowIdx, int subWarpColIdx)
    {
        float tmpA[WMITERS*TM];   
        float tmpB[WNITERS*TN];
        for(int idxK=0; idxK<BK; idxK++)
        {
            for(int witerM=0; witerM<WMITERS; witerM++)
            {
                for(int idxM=0; idxM<TM; idxM++)
                {
                    tmpA[witerM*TM + idxM] = MdS[idxK*BM + (warpRow*WM + witerM*WSUBM + subWarpRowIdx*TM+ idxM)]; //BM = 64, WM = 32, WSUBM =16, TM = 4. New tx at evry TM and every WN totally new
                }//Exchange row and col as we transposed.
            } 
            
            for(int witerN=0; witerN<WNITERS; witerN++)
            {
                for(int idxN = 0; idxN<TN; idxN++)
                {
                    tmpB[witerN*TN+ idxN] = NdS[idxK*BN + (warpCol*WN + witerN*WSUBN + subWarpColIdx*TN + idxN)];
                }
            }
            for(int witerM=0; witerM<WMITERS; witerM++)
            {
                for(int witerN=0; witerN<WNITERS; witerN++)
                {
                    for(int idxM=0; idxM<TM; idxM++)
                    {
                        for(int idxN=0; idxN<TN; idxN++)
                        {
                            pSum[(witerM*TM + idxM)*WNITERS*TN + (witerN*TN + idxN)] += (tmpA[witerM*TM + idxM]*tmpB[witerN*TN + idxN]);
                        }
                    }
                }
            }
        }
    }
}
__global__ void __launch_bounds__((BM*BN)/(TM*TN),1) matMulKernel(float* d_A, float* d_B, float* d_C, int M, int K, int N)
{

    const int innerRowA = threadIdx.x/(BK/4); //When we vectorize we do the Col Name/4 as float4
    const int innerColA = threadIdx.x%(BK/4); //Continue for all exp
    constexpr int rowStrideA = TX_PER_BLOCK/(BK/4); //This is for BK*BN > (4*Thread). We read in the BK(Col) globally. And add it to row here. Hence row stride

    const int innerRowB = threadIdx.x/(BN/4); //Same explation as 176
    const int innerColB = threadIdx.x%(BN/4);
    constexpr int rowStrideB = TX_PER_BLOCK/(BN/4); //Comparing it to BK (the row index as jump)


    const int outputsPerBlock = BM*BN; 
    const int outputsPerTile = TM*TN;

    assert(outputsPerBlock/outputsPerTile == blockDim.x); //As that many tx run in parallel during dot product phase. We need to compute.

    const int warpIdx = threadIdx.x/WARPSIZE; //Convert threadIdx to Warp
    const int warpRow = warpIdx/(BN/WN); //It spans BN with WN jump (It is between BN and WN width pixels)
    const int warpCol = warpIdx%(BN/WN);

    const int subWarpIdx = threadIdx.x%WARPSIZE; //This is with in Warp
    const int subWarpRowIdx = subWarpIdx/(WSUBN/TN); //WN is divided into WSUBNs. This spans WSUBN jumping TN.
    const int subWarpColIdx = subWarpIdx%(WSUBN/TN);


    d_A += blockIdx.y*BM*K; //Reading the Global input array is same
    d_B += blockIdx.x*BN; 
    d_C += ((blockIdx.y*BM + warpRow*WM)*N  + (blockIdx.x*BN + warpCol*WN)); //Output, each thread write a sub warp tile. Here we go the warp inside block.
    /*So, we are splitting BN into to 2 levels, 1. WNs and 2. Each WN into WSUBN. Then comes TN. Adjacent index in TN is sequentioal (same thread serially).
    Then the thread will jump WSUBN pixels (or WMITERS++). So, each WSUBM will have multiple TM. So, we will need WSUBN/TN tx. Then we do WNIRES++. BN/WN*WSUBN/TN threads total in Col *Row threads*/

    extern __shared__ float shMem[];
    
    float* MdS = shMem;
    float* NdS = &shMem[BM*BK];

    float pSum[TM*WMITERS*TN*WNITERS] = {0.0};
    
    for(int ph = 0; ph<K; ph+=BK)
    {
        //Load shared Mem
        warpKernels::loadFromGlobalToShared<rowStrideA, rowStrideB>(d_A, d_B, MdS, NdS, N, K, innerRowA, innerColA, innerRowB, innerColB); 
        __syncthreads();

        //Partial dot product
        warpKernels::dotProductSharedMem(MdS, NdS, pSum, warpRow, warpCol, subWarpRowIdx, subWarpColIdx);
        d_A += BK;
        d_B += (BK*N);
        __syncthreads();
        
    }
    
    for(int witerM=0; witerM<WMITERS; witerM++)
    {
        for(int witerN=0; witerN<WNITERS; witerN++)
        {
            for(int idxM=0; idxM<TM; idxM++)
            {
                for(int idxN=0; idxN<TN; idxN+=4)
                {
                    reinterpret_cast<float4*>(&d_C[(witerM*WSUBM + subWarpRowIdx*TM + idxM)*N + (witerN*WSUBN + subWarpColIdx*TN + idxN)])[0] = 
                                    reinterpret_cast<float4*>(&pSum[(witerM*TM + idxM)*WNITERS*TN + (witerN*TN + idxN)])[0]; //C is already shifted by necessary block and Warp. Now subWarp too.
                }//Same expression as in Psum dot product. Just split across different levels.
            }
        }
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
