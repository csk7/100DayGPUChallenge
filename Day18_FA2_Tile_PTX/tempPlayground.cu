#include<iostream>
#include<cuda.h>
#include<cuda_bf16.h>

typedef struct __nv_bfloat16 nv_bfloat16;
__global__ void bfloat16playGround(nv_bfloat16* dA, nv_bfloat16* dB, nv_bfloat16* dC, int N)
{
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx > N) return;
    extern __shared__ nv_bfloat16 shMem[];
    uint32_t PshMem = __cvta_generic_to_shared(shMem);

    uint32_t dst = PshMem + threadIdx.x*8*sizeof(nv_bfloat16);
    nv_bfloat16* src = dA + threadIdx.x*8;
    
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        :"r"(dst), "l"(src));

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    nv_bfloat16* PshMemPtr = reinterpret_cast<nv_bfloat16*>(__cvta_shared_to_generic(PshMem + threadIdx.x*8*sizeof(nv_bfloat16)));
    
    nv_bfloat162 tmpX = make_bfloat162(PshMemPtr[0], PshMemPtr[1]);
    nv_bfloat162 tmpY = make_bfloat162(PshMemPtr[2], PshMemPtr[3]);
    nv_bfloat162 tmpZ = make_bfloat162(PshMemPtr[4], PshMemPtr[5]);
    nv_bfloat162 tmpW = make_bfloat162(PshMemPtr[6], PshMemPtr[7]);
    reinterpret_cast<nv_bfloat162 *>(dC[threadIdx.x*8])[0] = tmpX;
    reinterpret_cast<nv_bfloat162 *>(dC[threadIdx.x*8+2])[0] = tmpY;
    reinterpret_cast<nv_bfloat162 *>(dC[threadIdx.x*8+4])[0] = tmpZ;
    reinterpret_cast<nv_bfloat162 *>(dC[threadIdx.x*8+6])[0] = tmpW;

    
    
}

int main()
{
    const int N=32;
    dim3 blocksPerKernel(1,1,1);
    dim3 threadsPerBlock(4,1,1);

    float* hA = new float[N];
    float* hB = new float[N];
    float* hC = new float[N];
    for(int i=0; i<N; i++)
    {
        hA[i] = i*2.5;
        hB[i] = i*2.5;
    }
    nv_bfloat16* bf16_hA = new nv_bfloat16[N];
    nv_bfloat16* bf16_hB = new nv_bfloat16[N];
    nv_bfloat16* bf16_hC = new nv_bfloat16[N];

    for(int i=0; i<N; i++)
    {
        bf16_hA[i] = __float2bfloat16(hA[i]);
        bf16_hB[i] = __float2bfloat16(hB[i]);
    }
    int size = N*sizeof(nv_bfloat16);
    
    nv_bfloat16* dA;
    nv_bfloat16* dB;
    nv_bfloat16* dC;

    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);

    cudaMemcpy(dA, bf16_hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, bf16_hB, size, cudaMemcpyHostToDevice);

    int shMemSize = N*sizeof(nv_bfloat16);

    bfloat16playGround<<<blocksPerKernel,threadsPerBlock, shMemSize>>>(dA, dB, dC, N);

    cudaMemcpy(bf16_hC, dC, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++)
    {
        hC[i] = __bfloat162float(bf16_hC[i]);
    }
    for(int i=0; i<N; i++)
    {
       printf("%f -- %f \n",hA[i], hC[i]);
    }

    printf("\n");

}