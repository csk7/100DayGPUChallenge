#include<cuda.h>
#include<cuda_bf16.h>

__device__ void globalToShared(uint32_t dstAddr, const nv_bfloat16* srcAddr, int numElementsPerLoad=8)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
        :
        : "r"(dstAddr), "l"(srcAddr), "n"(16));    
}