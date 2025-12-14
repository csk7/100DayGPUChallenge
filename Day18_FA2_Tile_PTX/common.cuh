#pragma once
#include<cuda.h>
#include<cuda_bf16.h>
#define CEIL_DIV(M,N) ((M)+(N)-1/(N))
__device__ void globalToShared(uint32_t dstAddr, const nv_bfloat16* srcAddr, int numElementsPerLoad=8)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
        :
        : "r"(dstAddr), "l"(srcAddr), "n"(16));    
}

template<const int HEIGHT, const int WIDTH>
__device__ __inline__ void globalToShared(uint32_t dstAddrBase, nv_bfloat16* srcAddrBase, int elemPerThread, int globalColDim = WIDTH)
{
    const int numThreads = blockDim.x;
    
    for(int iter=0; iter<((HEIGHT*WIDTH)/(numThreads*elemPerThread)); iter++)
    {
        const int index = iter*numThreads*elemPerThread + threadIdx.x*elemPerThread;
        const int rowIdx = index/WIDTH;
        const int colIdx = index%WIDTH;
        uint32_t dst = dstAddrBase + (rowIdx*WIDTH + colIdx)*sizeof(nv_bfloat16);
        nv_bfloat16* src = srcAddrBase + (rowIdx*globalColDim + colIdx);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(dst), "l"(src));
    }
}

__device__ __inline__ void sharedToRegx4(uint32_t regArray[4], const uint32_t srcShAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n" //x4 is number of tiles to load
        :"=r"(regArray[0]), "=r"(regArray[1]), "=r"(regArray[2]), "=r"(regArray[3])
        :"r"(srcShAddr));
}

__device__ __inline__ void sharedToRegx2(uint32_t regArray[2], const uint32_t srcAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%3];\n"
        :"=r"(regArray[0]), "=r"(regArray[1])
        :"r"(srcAddr));
}

__device__ __inline__ void sharedToRegx2Trans(uint32_t regs[2], const uint32_t srcAddr)
{
    asm volatile("ldmatrix.sync.aligned.m16n8.x2.trans.bf16 {%0, %1}; \n"
    :"=r"(regs[0]),"=r"(regs[1])
    :"r"(srcAddr))
}

__device__ __inline__ mma_m16n8k16(uint32_t D[4], uint32_t A[4], uint32_t B[2])
{
    asm volatile"("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    :"=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}