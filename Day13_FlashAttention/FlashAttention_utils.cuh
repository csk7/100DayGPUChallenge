#pragma once
#include<cuda.h>

#define CUDA_CHECK(call)          \
    do{                           \
        cudaError_t err = call;   \
        if(err != cudaSuccess)    \
            printf("Error %s at %d:",cudaGetErrorString(err),__LINE__); \
    }while(0);
                        

#define CEIL_CUSTOM(M, N) (((M) + (N) - 1)/(N))


template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ void __launch_bounds__((BM*BN)/(TM*TN),1) matMulKernel(float* d_A, float* d_B, float* d_C, int M, int K, int N)
{
    const int innerRowA = threadIdx.x/BK;
    const int innerColA = threadIdx.x%BK;
    const int strideA = blockDim.x/BK;

    const int innerRowB = threadIdx.x/BN;
    const int innerColB = threadIdx.x%BN;
    const int strideB = blockDim.x/BN;


    const int outputsPerBlock = BM*BN;
    const int outputsPerThread = TM*TN;

    assert(outputsPerBlock/outputsPerThread == blockDim.x);

    const int idxRow = threadIdx.x/(BN/TN);
    const int idxCol = threadIdx.x%(BN/TN);

    float pSum[TM][TN] = {0.0};
    float tmpA[TM];   
    float tmpB[TN];
    
    for(int ph = 0; ph<K; ph+=BK)
    {
        //Load shared Mem
        for(int iM = 0; iM<BM; iM+=strideA)
            MdS[(innerRowA+iM)*BK + innerColA] = d_A[(innerRowA+iM)*K + innerColA];
        for(int iN = 0; iN<BK; iN+=strideB)
            NdS[(innerRowB+iN)*BN + innerColB] = d_B[(innerRowB+iN)*N + innerColB];
        __syncthreads();
        

        d_A += BK;
        d_B += (BK*N);
        //Partial dot product
        for(int idxK=0; idxK<BK; idxK++)
        {
            for(int idxM=0; idxM<TM; idxM++)
            {
                tmpA[idxM] = MdS[(idxRow*TM + idxM)*BK + idxK];
            }
            for(int idxN = 0; idxN<TN; idxN++)
            {
                tmpB[idxN] = NdS[idxK*BN + (idxCol*TN+idxN)];
            }
            for(int idxM=0; idxM<TM; idxM++)
            {
                for(int idxN=0; idxN<TN; idxN++)
                {
                    pSum[idxM][idxN] += (tmpA[idxM]*tmpB[idxN]);
                }
            }
        }
        __syncthreads();
    }
    
    for(int idxM=0; idxM<TM; idxM++)
    {
        for(int idxN=0; idxN<TN; idxN++)
        {
            d_C[(idxRow*TM + idxM)*N + (idxCol*TN + idxN)] = pSum[idxM][idxN];
        }
    }
    
    
}