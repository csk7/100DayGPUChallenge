#pragma once
#include<iostream>
#include<cuda.h>
#include<cmath>

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
__device__ __inline__ T warpReduceSum(T val)
{
    return warpReduce<T>(val, __sum<T>);
}

template<typename T>
__device__ __inline__ T warpReduceMax(T val)
{
    return warpReduce<T>(val, __max<T>);
}

//BlockLevel Functions
template<typename T, typename Op, const int WARPSIZE=32>
__device__ __inline__ void blockReduce(T val, T* shMem, T emptyVal, int N, Op warpOp)
{
    const int tx = threadIdx.x;
    const int warpIdx = threadIdx.x/WARPSIZE;
    const int subWarpIdx = threadIdx.x%WARPSIZE;
    const int n_warps = (blockDim.x>N ? blockDim.x : N)  / WARPSIZE;

    val = warpOp(val);
    //Now 0, 32, 64 will have it. to 0,1,2, etc.
    
    if(subWarpIdx == 0)
        shMem[warpIdx] = val;
    __syncthreads();
    //Block level Max
    if(tx < WARPSIZE)
    {
        if(tx<n_warps)
            val = shMem[tx];
        else
            val = emptyVal;
        val = warpOp(val); 
    }

    if(tx == 0)
        shMem[0] = val; //Val is the own reg that was set eralier. It doesnt do anything to the sh mem.
    __syncthreads();
}

template<typename T>
__device__ __inline__ void blockReduceSum(T val, T* shMem, T emptyVal, int N)
{
    blockReduce<T>(val, shMem, emptyVal, N, warpReduceSum<T>);
}

template<typename T>
__device__ __inline__ void blockReduceMax(T val, T* shMem, T emptyVal, int N)
{
    blockReduce<T>(val, shMem, emptyVal, N, warpReduceMax<T>);
}