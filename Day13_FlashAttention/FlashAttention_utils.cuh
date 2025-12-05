#pragma once
#include<cuda.h>

#define CUDA_CHECK(call)          \
    do{                           \
        cudaError_t err = call;   \
        if(err != cudaSuccess)    \
            printf("Error %s at %d:",cudaGetErrorString(err),__LINE__); \
    }while(0);
                        

#define CEIL_CUSTOM(M, N) (((M) + (N) - 1)/(N))


template<const int BN=32>
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
    
    /*if(blockIdx.x == 1 && threadIdx.x == 0)
    {
        printf("GPU K: \n");
        for(int i=0; i<4; i++)
        {
            for(int j = 0; j<4; j++)
            {
                printf("%f \t", shK_T[i*BN + j]);
            }
            printf("\n");
        }
    }*/
}
template<const int BM=32>
__device__ void load_Q(float* Q, float* shQ, int N, int d)
{
    const int idxRow = threadIdx.x/d;
    const int idxCol = threadIdx.x%d;
    const int stride = blockDim.x/d;

    for(int i=0; i<BM; i+=stride)
    {
        shQ[(idxRow + i)*d + idxCol] =  Q[(idxRow + i)*d + idxCol];
    }
}

template<const int BM=32>
__device__ void load_O(float* O, float* shO, int d)
{
    assert(blockDim.x>=d);
    const int idxRow = threadIdx.x/d;
    const int idxCol = threadIdx.x%d;
    const int stride = blockDim.x/d;

    for(int i=0; i<BM; i+=stride)
    {
        shO[(idxRow + i)*d + idxCol] =  O[(idxRow + i)*d + idxCol];
    }
}

template<const int BM=32>
__device__ void load_L_i_M_i(float* shM_i, float* shL_i, float* m, float* l)
{
    assert(BM <= blockDim.x);
    
    if(threadIdx.x < BM)
    {
        shL_i[threadIdx.x] = l[threadIdx.x];
        shM_i[threadIdx.x] = m[threadIdx.x];
    }
}

template<const int BM=32, const int BN=32, const int TM=1, const int TN=1>
__device__ void __inline__ matMulS(float* shQ, float* shK_T, float* shS, int d)
{
    assert(BM*BN/(TM*TN) == blockDim.x);
    //To do dot product
    const int dotIdxRow = threadIdx.x/(BN/TN);
    const int dotIdxCol = threadIdx.x%(BN/TN);
    const int strideCol = blockDim.x/(BN/TN);
    const int strideRow = blockDim.x/(BM/TM);
    
    float regA[TM];
    float regB[TN];
    float pSum[TM][TN] = {0.0};

    for(int k=0; k<d; k++)
    {
        for(int i=0; i<TM; i++)
        {
            regA[i] = shQ[(dotIdxRow + i*strideRow)*d + k];
        }
        for(int i=0; i<TN; i++)
        {
            regB[i] = shK_T[k*BN + (dotIdxCol + i*strideCol)];
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


template<typename T, const int WARPSIZE=32, typename Op>
__device__ __inline__ T warpReduce(T val, Op warpOp)
{
    for(int level=WARPSIZE >> 1; level>=1; level >>= 1)
    {
        if(WARPSIZE == 32)
            val = warpOp(val,__shfl_down_sync(0xffffffff, val, level));
        else if(WARPSIZE == 4)
            val = warpOp(val,__shfl_down_sync(0xF, val, level));
        else 
            val = warpOp(val,__shfl_down_sync(0x3, val, level));
    }
    return val;
}

template<typename T, const int WARPSIZE=32>
__device__ __inline__ T warpSum(T val)
{
    return warpReduce<T, WARPSIZE>(val, __sum<T>);
}

template<typename T, const int WARPSIZE=32>
__device__ __inline__ T warpMax(T val)
{
    return warpReduce<T, WARPSIZE>(val, __max<T>);
}

//RowMax
template<const int BM=32, const int BN=32, const int TM=1, const int WARPSIZE=32>
__device__ void rowMax_calculateP_rowSum(float* shS_P, float* shM_ij, float* shL_ij)
{
    assert(BN <= WARPSIZE);
    const int idxRow = threadIdx.x/BN;
    const int idxCol = threadIdx.x%BN;
    const int stride = blockDim.x/BN;

    for(int i=0; i<TM; i++)
    {
        float valS_ij = shS_P[(idxRow + i*stride)*BN + idxCol];
        
        //Step 10;
        float valM_ij = warpMax<float, WARPSIZE>(valS_ij);
        if(idxCol == 0)
        {
            shM_ij[idxRow + i*stride] = valM_ij; //Getting the max wal of each row to all the threads in the warp. We have to go to shmem.
        }
        __syncthreads();
        valM_ij =  shM_ij[idxRow + i*stride];
        float tempP_ij = expf(valS_ij - valM_ij);
        shS_P[(idxRow + i*stride)*BN + idxCol] = tempP_ij;
        __syncthreads(); //TO-DO: Remove this.
        float valL_ij = warpSum<float, WARPSIZE>(tempP_ij);
        if(idxCol == 0)
        {
            shL_ij[idxRow + i*stride] = valL_ij; //Getting the sum across P of each row. This has to be propagated to multiple threads possible.
        }
        __syncthreads();
    }
    
}

//Step 11
template<const int BM=32>
__device__ void calculate_Mnew_i_Lnew_i(float* shM_i, float* shL_i, float* shM_ij, float* shM_New_i, float* shL_New_i, float* shL_ij)
{
    assert(BM <= blockDim.x);
    if(threadIdx.x < BM)
    {
        float valM_i  = shM_i[threadIdx.x];
        float valM_ij  = shM_ij[threadIdx.x];
        float valM_New_i = max(valM_ij, valM_i);

        shM_New_i[threadIdx.x] = valM_New_i;

        float valL_i = shL_i[threadIdx.x];
        float valL_ij = shL_ij[threadIdx.x];
        shL_New_i[threadIdx.x] = expf(valM_i - valM_New_i)*valL_i + expf(valM_ij - valM_New_i)*valL_ij;
    }
}

//Step12
template<const int BM=32, const int BN=32, const int BK=32, const int TM=1, const int TN=1>
__device__ void __inline__ matMulPV_Update_O(float* shV, float* shP, float* shO, \
                                                        float* shM_ij, float* shM_New_i, float* shM_i, float* shL_New_i, float* shL_i, int N, int d)
{
    assert(BM*BN/(TM*TN) == blockDim.x);
    //To do dot product
    const int dotIdxRow = threadIdx.x/(BN/TN);
    const int dotIdxCol = threadIdx.x%(BN/TN);
    const int strideCol = BN/TN;
    const int strideRow = BM/TM;

    float regA[TM];
    float regB[TN];
    float pSum[TM][TN] = {0.0};

    for(int k=0; k<BK; k++)
    {
        for(int i=0; i<TM; i++)
        {
            regA[i] = shP[(dotIdxRow + i*strideRow)*BK + k];
        }
        for(int i=0; i<TN; i++)
        {
            regB[i] = shV[k*BN + (dotIdxCol + i*strideCol)];
        }
        for(int i=0; i<TM; i++)
        {
            for(int j=0; j<TN; j++)
            {
                pSum[i][j] += regA[i]*regB[j];
            }
        }
    }

    float regM_New_i[TM];
    float regM_i[TM];
    float regM_ij[TM];

    float regL_New_i[TM];
    float regL_i[TM];
    for(int i=0; i<TM; i++)
    {
        //Load all M, and all L to registers
        regM_New_i[i] = shM_New_i[i*strideRow + dotIdxRow];
        regM_i[i] = shM_i[i*strideRow + dotIdxRow];
        regM_ij[i] = shM_ij[i*strideRow + dotIdxRow];

        regL_New_i[i] = shL_New_i[i*strideRow + dotIdxRow];
        regL_i[i] = shL_i[i*strideRow + dotIdxRow];

        for(int j=0; j<TN; j++)
        {
            shO[(dotIdxRow + i*strideRow)*d + (dotIdxCol + j*strideCol)] = ((regL_i[i]*expf(regM_i[i] - regM_New_i[i])*shO[(dotIdxRow + i*strideRow)*d + (dotIdxCol + j*strideCol)])
                                                        + (expf(regM_ij[i] - regM_New_i[i]) * pSum[i][j]))/regL_New_i[i]; 
            /*if(dotIdxRow == 0 && (dotIdxCol) == 2 && blockIdx.x == 0)
            {
                printf("shO at j=%d , why %d- is  : %f\n", dotIdxCol + j*strideCol, strideCol, shO[(dotIdxRow + i*strideRow)*d + (dotIdxCol + j*strideCol)]);
            }*/

        }
    }
}

template<const int BM=32>
__device__ void copy_L_i_M_i(float* shM_i, float* shL_i, float* shM_New_i, float* shL_New_i)
{
    assert(BM <= blockDim.x);
    if(threadIdx.x < BM)
    {
        shL_i[threadIdx.x] = shL_New_i[threadIdx.x];
        shM_i[threadIdx.x] = shM_New_i[threadIdx.x];
    }
}

template<const int BM=32>
__device__ void write_O(float* O, float* shO, int d)
{
    const int idxRow = threadIdx.x/d;
    const int idxCol = threadIdx.x%d;
    const int stride = blockDim.x/d;

    for(int i=0; i<BM; i+=stride)
    {
        O[(idxRow + i)*d + idxCol] =  shO[(idxRow + i)*d + idxCol];
        
    }
}