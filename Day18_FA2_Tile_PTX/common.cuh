#pragma once
#include <iostream>
#include <cstdint>

#include <cuda_bf16.h>
#define CEIL_DIV(M,N) (((M)+(N)-1)/(N))

template<const int STRIDE=128>
__device__ __inline__ uint32_t swizzle(uint32_t addr)
{
    if(STRIDE < 16) //Stride is [d*nv_bfloat16]
        return addr;

    const int row_index = (addr/STRIDE)%8; //Get row idx first from full address; 8 because we access 8x8 tiles. 8 rows. 0,1,2,3
    const int bits_to_xor = row_index/(max((64/STRIDE),1));//Large stride the row idex itself, means first 3 bits of row index[0-2] 
    return addr ^ (bits_to_xor << 4);//16B alignment so we shift by bits to 4
}

template<const int HEIGHT, const int WIDTH>
__device__ __inline__ void globalToSharedSwizzle(uint32_t dstAddrBase, const nv_bfloat16* srcAddrBase, int elemPerThread)
{
    const int numThreads = blockDim.x;    
    for(int iter=0; iter<((HEIGHT*WIDTH)/(numThreads*elemPerThread)); iter++)
    {
        const int index = iter*numThreads*elemPerThread + threadIdx.x*elemPerThread;
        const int rowIdx = index/WIDTH;
        const int colIdx = index%WIDTH;
        uint32_t dst = swizzle<WIDTH*sizeof(nv_bfloat16)>(dstAddrBase + (rowIdx*WIDTH + colIdx)*sizeof(nv_bfloat16));
        const nv_bfloat16* src = srcAddrBase + (rowIdx*WIDTH + colIdx);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(dst), "l"(src));
    }
}

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
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(dst), "l"(src));
    }
}


__device__ inline void sharedToRegx4(uint32_t regArray[4], uint32_t srcShAddr)
{
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

__device__ __inline__ void sharedToRegx4Trans(uint32_t regs[4], const uint32_t srcAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.trans.b16 {%0, %1, %2, %3}, [%4];\n"
    :"=r"(regs[0]),"=r"(regs[1]),"=r"(regs[2]),"=r"(regs[3])
    :"r"(srcAddr));
}

__device__ __inline__ void mma_m16n8k16(float D[4], uint32_t A[4], uint32_t B[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    :"=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}