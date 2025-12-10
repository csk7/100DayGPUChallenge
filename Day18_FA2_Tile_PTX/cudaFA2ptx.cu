#include<iostream>
#include<cassert>
#include<cuda.h>
#include<cuda_bf16.h>

#include"common.cuh"

/* Inputs are Q-> B, L, d and KV-> B, L, d and output is B, L, d*/

template<const int batchSize = 1, const int Br=128, const int Bc=128, const int d=128, const int numThreads = 128> 
__global__ void __launch_bounds__(numThreads) flashAttention2(const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V, nv_bfloat16* O, const int seqLength)
{
    const int row = blockIdx.x*Br + blockIdx.y*seqLength;
    if((blockIdx.x>batchSize) or ((blockIdx.z*Br)>seqLength))
    {
        return;
    }
    Q += row*d;
    K += (blockDim.y*seqLength)*d;
    V += (blockDim.y*seqLength)*d;
    O += row*d;

    extern __shared__ nv_bfloat16 shMem[];
    const uint32_t QshMem = __cvta_generic_to_shared(shMem);
    const uint32_t KshMem = QshMem + Br*d*sizeof(nv_bfloat16);
    const uint32_t VshMem = KshMem;
    //nv_bfloat16* OshMem = shMem + Br*d + 2*Bc*d;

    const int tid = threadIdx.x;
    //const int warpId = tid/WARPSIZE;
    //const int laneId = tid%WARPSIZE;
    //const int numWarps = blockDim.x/WARPSIZE; 
    
    //Global to shMem transfer
    //Load Q (Br*d)
    const int numElementsPerLoad = 8;
    assert((numElementsPerLoad*2) == 4 or (numElementsPerLoad*2) == 8 or (numElementsPerLoad*2) == 16);
    assert(numThreads >= (d/numElementsPerLoad));

    const int idxRow = tid/(d/numElementsPerLoad);
    const int idxCol = tid%(d/numElementsPerLoad);
    const int rowStride = blockDim.x/(d/numElementsPerLoad);
    printf("Hello \t");

    for(int iRows=0; iRows<Br ; iRows+=rowStride)
    {
        uint32_t dst = QshMem + ((idxRow+iRows)*d + idxCol*numElementsPerLoad)*sizeof(nv_bfloat16);
        const nv_bfloat16* src = Q + ((idxRow+iRows)*d + idxCol*numElementsPerLoad);
        printf("Q : %f \n", __bfloat162float(*src)); 
        globalToShared(dst, src, numElementsPerLoad);
    }
    

    //Load K (Bc*d)
    /*
    for(int iRows=0; iRows<Bc ; iRows+=rowStride)
    {
        uint32_t dst = KshMem + ((idxRow+iRows)*d + idxCol*numElementsPerLoad)*sizeof(nv_bfloat16);
        const nv_bfloat16* src = (const nv_bfloat16*)(&K[(idxRow+iRows)*d + idxCol*numElementsPerLoad]);
        globalToShared(dst, src, numElementsPerLoad);
    }

    //Load V
    for(int iRows=0; iRows<Bc; iRows+=rowStride)
    {
        uint32_t dst = VshMem + ((idxRow+iRows)*d + idxCol*numElementsPerLoad)*sizeof(nv_bfloat16);
        const nv_bfloat16* src = (const nv_bfloat16*)(&V[(idxRow+iRows)*d + idxCol*numElementsPerLoad]);
        globalToShared(dst, src, numElementsPerLoad);
    }
        */
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");

    //Write Qr back
    nv_bfloat16* QshMemPtr = reinterpret_cast<nv_bfloat16*>(QshMem);
    for(int iRows=0; iRows<Br; iRows+=rowStride)
    {
        reinterpret_cast<int4*>(&O[(idxRow+iRows)*d + idxCol*numElementsPerLoad])[0] = reinterpret_cast<int4*>(&QshMemPtr[(idxRow+iRows)*d + idxCol*numElementsPerLoad])[0];
    }


}

int main()
{
    const int batchSize=1;
    const int L=4;
    const int d=16;
    const int Br=4;
    const int Bc = 4;
    const int numThreads=4;
    dim3 blocksPerKernel(1,1,1);
    dim3 threadsPerBlock(numThreads,1,1);
    int shMemSize = (Br*d + 2*Bc*d)*sizeof(nv_bfloat16);

    float* hQ = new float[batchSize*L*d];
    float* hK = new float[batchSize*L*d];
    float* hV = new float[batchSize*L*d];
    float* hO = new float[batchSize*L*d];
    const int N = batchSize*L*d;
    for(int i=0; i<N; i++)
    {
        hQ[i] = i*2.5;
        hK[i] = i*2.5;
        hV[i] = i*2.5;
    }
    nv_bfloat16* bf16_hQ = new nv_bfloat16[N];
    nv_bfloat16* bf16_hK = new nv_bfloat16[N];
    nv_bfloat16* bf16_hV = new nv_bfloat16[N];
    nv_bfloat16* bf16_hO = new nv_bfloat16[N];

    for(int i=0; i<N; i++)
    {
        bf16_hQ[i] = __float2bfloat16(hQ[i]);
        bf16_hK[i] = __float2bfloat16(hK[i]);
        bf16_hV[i] = __float2bfloat16(hV[i]);
    }
    int size = N*sizeof(nv_bfloat16);
    
    nv_bfloat16* dQ;
    nv_bfloat16* dK;
    nv_bfloat16* dV;
    nv_bfloat16* dO;

    cudaMalloc((void**)&dQ, size);
    cudaMalloc((void**)&dK, size);
    cudaMalloc((void**)&dV, size);
    cudaMalloc((void**)&dO, size);

    cudaMemcpy(dQ, bf16_hQ, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, bf16_hK, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, bf16_hV, size, cudaMemcpyHostToDevice);

    flashAttention2<1, Br, Bc, d, numThreads><<<blocksPerKernel,threadsPerBlock, shMemSize>>>(dQ, dK, dV, dO, L);

    cudaMemcpy(bf16_hO, dO, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++)
    {
        hO[i] = __bfloat162float(bf16_hO[i]);
    }
    bool flag = true;
    for(int i=0; i<N; i++)
    {
        if(hQ[i] != hO[i])
        {
            printf("Error at loc %d, Input : %f != Output : %f\n", i, hQ[i], hO[i]);
            flag = false;
        }
    }
    if(flag == true)
    {
        printf("Success \n");
    }

    printf("\n");
}