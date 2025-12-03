#pragma once
#include<cuda.h>

#define CUDA_CHECK(call)          \
    do{                           \
        cudaError_t err = call;   \
        if(err != cudaSuccess)    \
            printf("Error %s at %d:",cudaGetErrorString(err),__LINE__); \
    }while(0);
                        

#define CEIL_CUSTOM(M, N) (((M) + (N) - 1)/(N))


template<const int BM, const int BN>
__device__ void load_K_V(float* K, float* V, float* shK_T, float* shV, int N, int d)
{
    const int idxRow = threadIdx.x/d;
    const int idxCol = threadIdx.x%d;
    const int stride = blockDim.x/d;

    for(int i=0; i<BN; i+=stride)
    {
        shK_T[idxCol*BN + (idxRow+i)] = K[(idxRow + i)*d + idxCol];
        shV[(idxRow + i)*d + idxCol] =  V[(idxRow + i)*d + idxCol];
    }
}


template<const int BM, const int BN, const int TM, const int TN>
__device__ void __launch_bounds__((BM*BN)/(TM*TN),1) __inline__ loadQ_matMulS(float* Q, float* shQ, float* shK_T, float** shS, int N, int d)
{
    assert(BM*BN/(TM*TN) == blockDim.x)
    //To store to shMem
    const int idxRow = threadIdx.x/d;
    const int idxCol = threadIdx.x%d;
    const int stride = blockDim.x/d;
    //To do dot product
    const int dotIdxRow = threadIdx.x/(BN/TN);
    const int dotIdxCol = threadIdx.x%(BN/TN);
    const int strideCol = blockDim.x/(BN/TN);
    const int strideRow = blockDim.x/(BM/TM);
    

    //load Shared Mem
    for(int i=0; i<BM; i+=stride)
    {
        shQ[(idxRow + i)*BK + idxCol] = Q[(idxRow + i)*d + idxCol];
    }
    __syncthreads();
    float regA[TM];
    float regB[TN];
    float pSum[TM][TN];

    for(int k=0; k<d; k++)
    {
        for(int i=0; i<TM; i++)
        {
            regA[i] = shQ[(dotIdxRow + i*strideRow)*d + k];
        }
        for(int i=0; i<TN; i++)
        {
            regB[i] = shB[k*BN + (dotIdxCol + i*strideCol)];
        }
        for(int i=0; i<TM; i++)
        {
            for(int j=0; j<TN; j++)
            {
                pSum[i][j] += regA[i]*regB[j];
            }
        }
    }

    for(int i=0; i<TM; i++)
    {
        for(int j=0; j<TN; j++)
        {
            shS[(dotIdxRow + i*strideRow)*BN + (dotIdxCol + j*strideCol)] = pSum[i][j];
        }
    }
  
}

//Warp Level Functions 
template<typename T>
__device__ __forceinline__ T __max(T value1, T value2)
{
    return max(value1, value2);
}

template<typename T>
__device__ __forceinline__ T __sum(T value1, T value2)
{
    return (value1 + value2);
}


template<typename T, typename Op, const int WARPSIZE=32>
__device__ __inline__ T warpReduce(T val, Op warpOp)
{
    for(int level=WARPSIZE >> 1; level>=1; level >>= 1)
    {
        val = warpOp(val,__shfl_down_sync(0xffffffff, val, level));
    }
    return val;
}

template<typename T>
__device__ __inline__ T warpSum(T val)
{
    return warpReduce<T>(val, __sum<T>);
}

template<typename T>
__device__ __inline__ T warpMax(T val)
{
    return warpReduce<T>(val, __max<T>);
}

//RowMax
template<const int BM, const int BN, const int TM, const int WARPSIZE=32>
__device__ void rowMax_calculateP_rowSum(float* shS_P, float* shM_ij, float* shL_ij)
{
    assert(BN == WARPSIZE);
    const int idxRow = threadIdx.x/BN;
    const int idxCol = threadIdx.x%BN;
    const int stride = blockDim.x/BN;

    for(int i=0; i<TM; i++)
    {
        float valS_ij = shS_P[(idxRow + i*stride)*BN + idxCol];
        //Step 10;
        float valM_ij = warpSum(valMax);
        
        if(idxCol == 0)
        {
            shM_ij[idxRow + i*stride] = valM_ij; //Getting the max wal of each row to all the threads in the warp. We have to go to shmem.
        }
        __syncthreads();
        valM_ij =  shM_ij[idxRow + i*stride];
        float tempP_ij = expf(valS_ij - valM_ij);
        shS[(idxRow + i*stride)*BN + idxCol] = tempP_ij;
        __syncthreads(); //TO-DO: Remove this.
        valL_ij = warpSum(tempP_ij);
        if(idxCol == 0)
        {
            shL_ij[idxRow + i*stride] = valL_ij; //Getting the sum across P of each row. This has to be propagated to multiple threads possible.
        }
        __syncthreads();
    }
    
}

template<const int BM, const int WARPSIZE=32>
__device__ void calculate_Mnew_i_Lnew_i(float* shM_i, float* shL_i, float* shM_ij, float* shL_ij)
{
    assert(BM < blockDim.x)
    if(threadIdx.x < BM)
    {
        float valM_i  = shM_i[idxRow + i*stride];
        float valM_New_i = max(valM_ij, valM_i);

        float valL_i  = shL_i[idxRow + i*stride];
        float valL_New_i = exp(valM_i, valM_New_i)*valL_i + exp(valM_ij, valM_New_i)*valL_ij;
    }
}