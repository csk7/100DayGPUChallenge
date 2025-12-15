#include<iostream>
#include<cuda_bf16.h>

using namespace std;

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


__device__ inline void sharedToRegx4(int regArray[4], uint32_t srcShAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" //x4 is number of tiles to load
        :"=r"(regArray[0]), "=r"(regArray[1]), "=r"(regArray[2]), "=r"(regArray[3])
        :"r"(srcShAddr));
}

__device__ __inline__ void sharedToRegx2(int regArray[2], const uint32_t srcAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        :"=r"(regArray[0]), "=r"(regArray[1])
        :"r"(srcAddr));
}

__device__ __inline__ void mma_m16n8k16(float D[4], int A[4], int B[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    :"=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

template<const int Br=128, const int Bc=128, const int d = 128, const int WARPSIZE=32> 
__global__ void flashAttention2(const nv_bfloat16* Q, const nv_bfloat16* K, float* C)
{
    //Shared Mem declaration
    __shared__ alignas(16) nv_bfloat16 QshMem[Br*d];
    __shared__ alignas(16) nv_bfloat16 KshMem[Bc*d];

    const int laneId = threadIdx.x%WARPSIZE;

    //Reg declaration
    nv_bfloat16 Qreg[8]; 
    nv_bfloat16 Kreg[4];


    //Start - Load Q
    if(threadIdx.x<16)
    {
        for(int i =0; i<16; i++)
        {       
            QshMem[threadIdx.x*d + i] = Q[threadIdx.x*d + i];
        }
    }
    __syncthreads();

    
    if(threadIdx.x == 0)
        printf("In function Reg by thread %d is {%d} \n", threadIdx.x, 1);

    if(threadIdx.x<8)
    {
        for(int i =0; i<16;i++)
        { 
           KshMem[threadIdx.x*d + i] = K[threadIdx.x*d + i];
        }
    }
    __syncthreads();
    if(threadIdx.x == 0)
        printf("In function Reg by thread %d is {%d} \n", threadIdx.x, 2);

    int *QregInt = (int *)Qreg;
    int *KregInt = (int *)Kreg;

    const uint32_t srcShAddressQ = __cvta_generic_to_shared(&QshMem[(laneId%16)*d + ((laneId/16)*8)]);
    sharedToRegx4(QregInt, srcShAddressQ);

    __syncthreads();
    if(threadIdx.x == 0)
        printf("In function Reg by thread %d is {%d} \n", threadIdx.x, 3);

    const uint32_t KshMemInt = __cvta_generic_to_shared(&KshMem[0]);
    //Shared to Reg
    const uint32_t srcShAddressK= KshMem[((laneId%8)*d + (laneId/8*8))];
    sharedToRegx2(KregInt, srcShAddressK);


    __syncthreads();

    if(threadIdx.x == 0)
        printf("In function Reg by thread %d is {%d} \n", threadIdx.x, 4);

    float Sreg[4] = {};
    mma_m16n8k16(Sreg, QregInt, KregInt);

    const int idxRow = threadIdx.x/4;
    const int idxCol = (threadIdx.x%4)*2;
    
    C[idxRow*8 + idxCol] = Sreg[0];
    C[idxRow*8 + (idxCol+1)] = Sreg[1];
    C[(idxRow+8)*8 + idxCol] = Sreg[2];
    C[(idxRow+8)*8 + (idxCol+1)] = Sreg[3];
    

    //if(threadIdx.x == 0)
    //    printf("In function Reg by thread %d is {%.3f, %.3f, %.3f, %.3f} \n", threadIdx.x, Sreg[0], Sreg[1], Sreg[2], Sreg[3]);


}


void flashAttention2_v1(const nv_bfloat16* Q, const nv_bfloat16* K, float* C)
{
    // Get CUDA runtime and driver versions
    int runtimeVersion = 0;
    int driverVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    const int Br = 16;
    const int Bc = 8;
    const int d = 16;
    const int WARPSIZE = 32;

    dim3 threadsPerBlock(32, 1, 1);
    dim3 blocksPerKernel(1, 1, 1);

    printf("Hello Cu Calling \n");

    cudaDeviceSynchronize();
    flashAttention2<Br, Bc, d, WARPSIZE><<<blocksPerKernel, threadsPerBlock>>>(Q, K, C);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

}


int main()
{
    const int seqLengthQ=16;
    const int seqLengthKV=8;
    const int d = 16;

    float* hQ = new float[seqLengthQ*d];
    float* hK = new float[seqLengthKV*d];
    float* hO = new float[seqLengthQ*seqLengthKV];
    const int NQ = seqLengthQ*d;
    const int NKV = seqLengthKV*d;
    const int NO = seqLengthQ*seqLengthKV;
    for(int i=0; i<NQ; i++)
    {
        hQ[i] = i*2.0;
    }
    for(int i=0; i<NKV; i++)
    {
        hK[i] = i*2.0;
    }
    for(int i=0; i<NO; i++)
    {
        hO[i] = 0.0;
    }
    nv_bfloat16* bf16_hQ = new nv_bfloat16[NQ];
    nv_bfloat16* bf16_hK = new nv_bfloat16[NKV];


    for(int i=0; i<NQ; i++)
    {
        bf16_hQ[i] = __float2bfloat16(hQ[i]);

    }
    for(int i=0; i<NKV; i++)
    {
        bf16_hK[i] = __float2bfloat16(hK[i]);

    }
    int sizeQ = NQ*sizeof(nv_bfloat16);
    int sizeKV = NKV*sizeof(nv_bfloat16);
    int sizeO = NO*sizeof(nv_bfloat16);
    
    nv_bfloat16* dQ;
    nv_bfloat16* dK;
    float* dO;
    cudaMalloc((void**)&dQ, sizeQ);
    cudaMalloc((void**)&dK, sizeKV);
    cudaMalloc((void**)&dO, sizeO);
    cudaMemcpy(dQ, bf16_hQ, sizeQ, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, bf16_hK, sizeKV, cudaMemcpyHostToDevice);
    
    flashAttention2_v1(reinterpret_cast<const nv_bfloat16*>(dQ), reinterpret_cast<const nv_bfloat16*>(dK), dO);
    cudaMemcpy(hO, dO, sizeO, cudaMemcpyDeviceToHost);
    
    printf("Finished Cuda call\n");

}
