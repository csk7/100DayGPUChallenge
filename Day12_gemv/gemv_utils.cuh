#pragma once
#include<cuda.h>

template<typename T, typename Op, const int WARPSIZE = 32> 
__device__ __inline__ T warpReduce(T regVal, Op op)
{
    for(int stride = WARPSIZE/2; stride>=1; stride/=2)
    {
        regVal = op(regVal, __shfl_down_sync(0xffffffff, regVal, stride));
    }
    return regVal;
}

template<typename T>
__device__ __inline__ T __opSum(T inputA, T inputB)
{
    return (inputA + inputB);
}

template<typename T> 
__device__ __inline__ T warpSum(T regVal)
{
    return warpReduce<float>(regVal, __opSum<T>);
}

template<typename T, typename Op, const int WARPSIZE = 32> 
__device__ void blockReduce(T regVal, T* shMem, T emptyVal, Op warpOp, int BN)
{
    const int tid = threadIdx.x;
    const int idxRow = threadIdx.x/BN;
    const int idxCol = threadIdx.x%BN;

    const int nWarpsPerRow = BN/WARPSIZE;

    regVal = warpOp(regVal);
    
    if(tid%WARPSIZE == 0)
    {
        shMem[idxRow*nWarpsPerRow + idxCol/WARPSIZE] = regVal;
    }
    __syncthreads();

    if(idxCol<WARPSIZE)
    {
        if(idxCol < nWarpsPerRow)
            regVal =  shMem[idxRow*nWarpsPerRow + idxCol];
        else    
            regVal = emptyVal;

        regVal = warpOp(regVal);
    }
    if(idxCol == 0)
    {
        shMem[idxRow*nWarpsPerRow] = regVal;
    }
    __syncthreads();

}

template<typename T> 
__device__ void blockSum(T regVal, T* shMem, T emptyVal, int BN)
{
    blockReduce<T>(regVal, shMem, emptyVal, warpSum<T>, BN);
}