#pragma once
#include <iostream>
#include <cstdint>

#include <cuda_bf16.h>
#define CEIL_DIV(M,N) (((M)+(N)-1)/(N))
/*__device__ void globalToShared(uint32_t dstAddr, const nv_bfloat16* srcAddr, int numElementsPerLoad=8)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
        :
        : "r"(dstAddr), "l"(srcAddr), "n"(16));    
}*/

template<const int HEIGHT, const int WIDTH>
__device__ __inline__ void globalToShared(uint32_t dstAddrBase, const nv_bfloat16* srcAddrBase, int elemPerThread)
{
    const int numThreads = blockDim.x;    
    for(int iter=0; iter<((HEIGHT*WIDTH)/(numThreads*elemPerThread)); iter++)
    {
        const int index = iter*numThreads*elemPerThread + threadIdx.x*elemPerThread;
        const int rowIdx = index/WIDTH;
        const int colIdx = index%WIDTH;
        uint32_t dst = dstAddrBase + (rowIdx*WIDTH + colIdx)*sizeof(nv_bfloat16);
        const nv_bfloat16* src = srcAddrBase + (rowIdx*WIDTH + colIdx);
        //printf("Read Elements by thread : %d is {%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f} \n",threadIdx.x, 
        //    __bfloat162float(src[0]), __bfloat162float(src[1]), __bfloat162float(src[2]), __bfloat162float(src[3]),
        //    __bfloat162float(src[4]), __bfloat162float(src[5]), __bfloat162float(src[6]), __bfloat162float(src[7])
        //);
        //printf("Iters : %d, blockIdx : %d , Dest by thread %d is %d \n", iter, blockIdx.x, threadIdx.x, dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(dst), "l"(src));
    }
}


__device__ inline void sharedToRegx4(uint32_t regArray[4], uint32_t srcShAddr)
{

    /*nv_bfloat162 tempPrint1 = reinterpret_cast<nv_bfloat162*>(__cvta_shared_to_generic(srcShAddr + (0)*sizeof(nv_bfloat162)))[0];
    nv_bfloat162 tempPrint2 = reinterpret_cast<nv_bfloat162*>(__cvta_shared_to_generic(srcShAddr + (1)*sizeof(nv_bfloat162)))[0];
    nv_bfloat162 tempPrint3 = reinterpret_cast<nv_bfloat162*>(__cvta_shared_to_generic(srcShAddr + (2)*sizeof(nv_bfloat162)))[0];
    nv_bfloat162 tempPrint4 = reinterpret_cast<nv_bfloat162*>(__cvta_shared_to_generic(srcShAddr + (3)*sizeof(nv_bfloat162)))[0];
    float2 tempPrintFloat1 = __bfloat1622float2(tempPrint1);
    float2 tempPrintFloat2 = __bfloat1622float2(tempPrint2);
    float2 tempPrintFloat3 = __bfloat1622float2(tempPrint3);
    float2 tempPrintFloat4 = __bfloat1622float2(tempPrint4);
    printf("thread Val : %d with address %d with vals: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", threadIdx.x, srcShAddr, tempPrintFloat1.x, tempPrintFloat1.y, tempPrintFloat2.x, tempPrintFloat2.y,
        tempPrintFloat3.x, tempPrintFloat3.y, tempPrintFloat4.x, tempPrintFloat4.y);*/

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" //x4 is number of tiles to load
        :"=r"(regArray[0]), "=r"(regArray[1]), "=r"(regArray[2]), "=r"(regArray[3])
        :"r"(srcShAddr));
    
}

__device__ __inline__ void sharedToRegx2(uint32_t regArray[2], const uint32_t srcAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        :"=r"(regArray[0]), "=r"(regArray[1])
        :"r"(srcAddr));
}

__device__ __inline__ void sharedToRegx2Trans(uint32_t regs[2], const uint32_t srcAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 {%0, %1}, [%2];\n"
    :"=r"(regs[0]),"=r"(regs[1])
    :"r"(srcAddr));
}

__device__ __inline__ void mma_m16n8k16(float D[4], uint32_t A[4], uint32_t B[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    :"=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}