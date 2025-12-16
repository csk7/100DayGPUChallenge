#include<iostream>
#include<cuda_bf16.h>
#include<cstdint>
#include<float.h>



#include"common.cuh"

using namespace std;

/* Inputs are Q-> B, L, d and KV-> B, L, d and output is B, L, d*/

template<const int Br=128, const int Bc=128, const int d=128, const int WARPSIZE=32, const int NUMTHREADS=128> 
__global__ void flashAttention2(const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V, nv_bfloat16* O, const int seqLength, int batchSize)
{
    const int row = blockIdx.x*Br + blockIdx.y*seqLength;
    if((blockIdx.x>batchSize) or ((blockIdx.z*Br)>seqLength))
    {
        return;
    }
    //Block pointer moved
    Q += row*d;
    //K += (blockDim.y*seqLength)*d;
    //V += (blockDim.y*seqLength)*d;
    O += row*d;

    //Shared Mem declaration
    extern __shared__ nv_bfloat16 shMem[];
    const uint32_t QshMem = __cvta_generic_to_shared(shMem);
    const uint32_t KshMem = QshMem;
    const uint32_t VshMem = KshMem +  Bc*d*sizeof(nv_bfloat16);

    //Thread Index to Warp Id and lane Id
    const int tid = threadIdx.x;
    const int warpId = tid/WARPSIZE;
    const int laneId = tid%WARPSIZE;
    const int numWarps = NUMTHREADS/WARPSIZE; 
    const int blockQperWarp = Br/numWarps;

    //Tiled MMA size
    const int MMA_M = 16;
    const int MMA_K = 16;
    const int MMA_N = 8;

    //Reg declaration
    uint32_t Qreg[blockQperWarp/MMA_M][d/MMA_K][4]; //4  16*16/32 following below logic (2  bf16 per reg)
    uint32_t Kreg[Bc/MMA_N][d/MMA_K][2];
    uint32_t Vreg[Bc/MMA_K][d/MMA_N][2];
    uint32_t Preg[Bc/MMA_K][d/MMA_N][4];
    //float Sreg[blockQperWarp/MMA_M][Bc/MMA_N][4]; //fp32. So per warp total space/32 --> 16*8/32
    float Oreg[blockQperWarp/MMA_M][d/MMA_K][4] = {}; //4  is number of tiles per warp

    float rowMax[blockQperWarp/MMA_M][2]; 
    for(int i=0; i<(blockQperWarp/MMA_M); i++)
    {
        rowMax[i][0] = -INFINITY;
        rowMax[i][1] = -INFINITY;
    }

    float rowSumExp[blockQperWarp/MMA_M][2];
    for(int i=0; i<(blockQperWarp/MMA_M); i++)
    {
        rowSumExp[i][0] = 0.0;
        rowSumExp[i][1] = 0.0;
    }

    const float scaling_factor = rsqrtf(static_cast<float>(d));
    //Global to shMem transfer
    //Load Q (Br*d)
    const int numElementsPerLoad = 8;

    //Start - Load Q
    globalToShared<Br, d>(QshMem, Q, numElementsPerLoad);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    
    for(int mma_id_row=0; mma_id_row<(blockQperWarp/MMA_M); mma_id_row++)
    {
        for(int mma_id_col=0; mma_id_col<(d/MMA_K); mma_id_col++)
        {
            const int idxRow = warpId*blockQperWarp + mma_id_row*MMA_M + laneId%16; //16 is a function of how many tiles (rowWise) are there in the m8n8 load matrix. 2 tiles(2*m8) so 16
            const int idxCol = mma_id_col*MMA_K + (laneId/16)*8; //16 --> from above, ldMatrix m8n8. n8 contributes to this;
            const uint32_t srcShAddress = QshMem + (idxRow*d + idxCol)*sizeof(nv_bfloat16);
            sharedToRegx4(Qreg[mma_id_row][mma_id_col], srcShAddress);
        }
    }
    
    __syncthreads();


    for(int idx_KV = 0; idx_KV<seqLength; idx_KV+=Bc)
    {
        float Sreg[blockQperWarp/MMA_M][Bc/MMA_N][4] = {};
        //Load K (Bc*d), Global to Shared
        globalToShared<Bc, d>(KshMem, K, numElementsPerLoad);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();
        

        //Shared to Reg
        for(int mma_id_row=0; mma_id_row<Bc; mma_id_row+=MMA_N)
        {
            for(int mma_id_col=0; mma_id_col<d; mma_id_col+=MMA_K)
            {
                const int idxRow = mma_id_row + laneId%8; //8 is a function of how many tiles (rowWise) are there in the [m8]n8 load matrix
                const int idxCol = mma_id_col + laneId/8*8; //Second 8 is a function of m8[n8] load matrix
                const uint32_t srcShAddress = KshMem + (idxRow*d + idxCol)*sizeof(nv_bfloat16);
                sharedToRegx2(Kreg[mma_id_row/MMA_N][mma_id_col/MMA_K], srcShAddress);
            }
        }
        
        __syncthreads();

        //MMA
        for(int i=0; i<(blockQperWarp/MMA_M); i++)
        {
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                for(int k=0; k<(d/MMA_K); k++)
                {
                    mma_m16n8k16(Sreg[i][j], Qreg[i][k], Kreg[j][k]);
                }
            }
        }
        __syncthreads();
        //printf("Row,Col %d,%d , %.3f, %.3f\n",threadIdx.x/4, (threadIdx.x%4)*2+8, Sreg[0][1][0], Sreg[0][1][1]);
        //printf("Row,Col %d,%d , %.3f, %.3f\n",(threadIdx.x/4) + 8, (threadIdx.x%4)*2+8, Sreg[0][1][2], Sreg[0][1][3]);


        //Head scaling and Row max
        for(int i=0; i<(blockQperWarp/MMA_M); i++)
        {
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                for(int k=0; k<4; k++)
                {
                    Sreg[i][j][k] *= scaling_factor;
                }
            }
            //row max, online softmax
            float maxTileRow[2] = {-INFINITY, -INFINITY};
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                float* tempReg = Sreg[i][j];
                maxTileRow[0] = max(maxTileRow[0], max(tempReg[0], tempReg[1]));
                maxTileRow[1] = max(maxTileRow[1], max(tempReg[2], tempReg[3]));
            } //We close here as we get all the values pertianing to this row. No need separately. Clever

            maxTileRow[0] = max(maxTileRow[0], __shfl_xor_sync(0xffffffff, maxTileRow[0], 1));
            maxTileRow[0] = max(maxTileRow[0], __shfl_xor_sync(0xffffffff, maxTileRow[0], 2));
            maxTileRow[1] = max(maxTileRow[1], __shfl_xor_sync(0xffffffff, maxTileRow[1], 1));
            maxTileRow[1] = max(maxTileRow[1], __shfl_xor_sync(0xffffffff, maxTileRow[1], 2));

            __syncthreads();

            maxTileRow[0] = max(maxTileRow[0], rowMax[i][0]);
            maxTileRow[1] = max(maxTileRow[1], rowMax[i][1]);

            
            __syncthreads();
            //rescale
            float rescale[2];
            rescale[0] = __expf(rowMax[i][0] - maxTileRow[0]);
            rescale[1] = __expf(rowMax[i][1] - maxTileRow[1]);

            for(int j=0; j<(Bc/MMA_N); j++)
            {
                Oreg[i][j][0] *= rescale[0];
                Oreg[i][j][1] *= rescale[0];
                Oreg[i][j][2] *= rescale[1];
                Oreg[i][j][3] *= rescale[1];
            }

            //copy max val for next MMN_N tile
            rowMax[i][0] = maxTileRow[0];
            rowMax[i][1] = maxTileRow[1];

            //RowSumExp + Need to write P as uint32_t and not float in S. Also m16n8 needs to be converted to m16n16
            float sumexp[2] = {0.0, 0.0};
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                Sreg[i][j][0] = __expf(Sreg[i][j][0] - rowMax[i][0]);
                Sreg[i][j][1] = __expf(Sreg[i][j][1] - rowMax[i][0]);
                Sreg[i][j][2] = __expf(Sreg[i][j][2] - rowMax[i][1]);
                Sreg[i][j][3] = __expf(Sreg[i][j][3] - rowMax[i][1]);

                //row sum in same tx
                sumexp[0] += Sreg[i][j][0] + Sreg[i][j][1];
                sumexp[1] += Sreg[i][j][2] + Sreg[i][j][3];

                //Need to rearange P
                nv_bfloat162* this_Preg = reinterpret_cast<nv_bfloat162*>(Preg[i][j/2]); //We need to accumulate 2 n/8 to n/16. So 0 and 1 S points to same P
                this_Preg[(j%2)*2] = __float22bfloat162_rn({Sreg[i][j][0], Sreg[i][j][1]}); //P has 4 reg to fill. 0 and 1 filled when j is even and 2 and 3 when its odd [As Zig Zag orders x2 to x4]
                this_Preg[(j%2)*2+1] = __float22bfloat162_rn({Sreg[i][j][2], Sreg[i][j][3]});
            }

            if(threadIdx.x == 0)
                printf("Hello10 \n");

            //Rwo sum across tx
            sumexp[0] += __shfl_xor_sync(0xffffffff, sumexp[0], 1);
            sumexp[0] += __shfl_xor_sync(0xffffffff, sumexp[0], 2);
            sumexp[1] += __shfl_xor_sync(0xffffffff, sumexp[1], 1);
            sumexp[1] += __shfl_xor_sync(0xffffffff, sumexp[1], 2);

            __syncthreads();

            if(threadIdx.x == 0)
                printf("Hello11 \n");
            

            rowSumExp[i][0] = rowSumExp[i][0]*rescale[0] + sumexp[0];
            rowSumExp[i][1] = rowSumExp[i][1]*rescale[1] + sumexp[1];
            if(threadIdx.x%4 == 0)
            {
                printf("Max of Row %d is %.3f\n",threadIdx.x/4, rowSumExp[i][0]);
                printf("Max of Row %d is %.3f\n",(threadIdx.x/4) + 8, rowSumExp[i][1]);
            }

            __syncthreads();

            if(threadIdx.x == 0)
                printf("Hello12 \n");
        }

        //Load V to registers
        globalToShared<Bc, d>(VshMem, V, numElementsPerLoad);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();
        /*if(threadIdx.x == 0 and blockIdx.y == 0 and blockIdx.x == 0)
        {
            for(int i=0; i<16; i++)
            {
                for(int j=0; j<8; j++)
                {
                    nv_bfloat162 tempPrint = reinterpret_cast<nv_bfloat162*>(__cvta_shared_to_generic(VshMem + (i*8 + j)*sizeof(nv_bfloat162)))[0];
                    float2 tempPrintFloat = __bfloat1622float2(tempPrint);
                    printf("%.3f, %.3f, ", tempPrintFloat.x, tempPrintFloat.y);
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        if(threadIdx.x == 0)
                printf("Hello13 \n");

        for(int mma_i_row=0; mma_i_row<(Bc/MMA_K); mma_i_row++)
        {
            for(int mma_j_col=0; mma_j_col<(d/MMA_N); mma_j_col++)
            {
                const int idxRow = mma_i_row*MMA_K + laneId%16;
                const int idxCol = mma_j_col*MMA_N + laneId/16*8;
                const uint32_t srcAddr = VshMem + (idxRow*d + idxCol)*sizeof(nv_bfloat16);
                sharedToRegx2Trans(Vreg[mma_i_row][mma_j_col], srcAddr);
            }
        }
        __syncthreads();

        if(threadIdx.x == 0)
                printf("Hello14 \n");

        //MMA
        for(int i=0; i<(blockQperWarp/MMA_M); i++)
        {
            for(int j=0; j<(d/MMA_N); j++)
            {
                for(int k=0; k<(Bc/MMA_K); k++)
                {
                    mma_m16n8k16(Oreg[i][j], Preg[i][k], Vreg[k][j]);
                }
            }
        }
        __syncthreads();

        if(threadIdx.x == 0)
                printf("Hello15 \n");



        K += Bc*d;
        V += Bc*d;
        __syncthreads();
    }

    for(int mma_id_row=0; mma_id_row<(blockQperWarp/MMA_M); mma_id_row++)
    {
        for(int mma_id_col=0; mma_id_col<(d/MMA_N); mma_id_col++)
        {
            const int idxRow = warpId*blockQperWarp + mma_id_row*MMA_M + laneId/4; //So, 32 tx write a m16n8 output. So each tx writes 4 outs. fp32 out each occupy 1 reg.
            //Tx 0 gets 4 outputs. Row0, Col0, Col1 stored in Reg0 and Reg1 as fp32. If it was bf16 then Both in reg0. 
            //Only half the ouput regs are written by a tx in one row. Other half to m16/2 row. In fp32 you need 4 tx to fill a row. 
            //Each tx rows are current row + m16/2.   
            const int idxCol = mma_id_col*MMA_N + (laneId%4)*2; //2 as each row is writing 2 elems. 8 n8 so %4.  
            
            //scaling softmax sum
            float* temp = Oreg[mma_id_row][mma_id_col];
            temp[0] = temp[0]/rowSumExp[mma_id_row][0];
            temp[1] = temp[1]/rowSumExp[mma_id_row][0];
            temp[2] = temp[2]/rowSumExp[mma_id_row][1];
            temp[3] = temp[3]/rowSumExp[mma_id_row][1];
            
            reinterpret_cast<nv_bfloat162*>(&O[idxRow*d + idxCol])[0] = __float22bfloat162_rn({temp[0], temp[1]});
            reinterpret_cast<nv_bfloat162*>(&O[(idxRow + 8)*d + idxCol])[0] = __float22bfloat162_rn({temp[2], temp[3]});
        }
    }

    if(threadIdx.x == 0)
        printf("Hello16 \n");


}


void flashAttention2_v1(const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V, nv_bfloat16* O, int seqLength, int batchSize)
{
    // Get CUDA runtime and driver versions
    int runtimeVersion = 0;
    int driverVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);

    
    const int Br = 16;
    const int Bc = 16;
    const int d = 16;
    const int numThreads = 32;
    const int WARPSIZE = 32;

    dim3 threadsPerBlock(numThreads, 1, 1);
    dim3 blocksPerKernel(CEIL_DIV(seqLength, Br), batchSize, 1);
    const int shMemSize = max(Br, 2*Bc)*d*sizeof(nv_bfloat16);

    printf("Hello Cu Calling %d\n",batchSize);

    cudaDeviceSynchronize();
    flashAttention2<Br, Bc, d, WARPSIZE, numThreads><<<blocksPerKernel, threadsPerBlock, shMemSize>>>(Q, K, V, O, seqLength, batchSize);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess)
    {
        printf("Error : %s\n",cudaGetErrorString(err));
    }

}


int main()
{
    const int batchSize=1;
    const int seqLength=16;
    const int d = 16;

    float* hQ = new float[batchSize*seqLength*d];
    float* hK = new float[batchSize*seqLength*d];
    float* hV = new float[batchSize*seqLength*d];
    float* hO = new float[batchSize*seqLength*d];
    const int N = batchSize*seqLength*d;
    for(int i=0; i<N; i++)
    {
        hQ[i] = i*2.0;
        hK[i] = i*2.0;
        hV[i] = i*2.0;
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

    flashAttention2_v1(reinterpret_cast<const nv_bfloat16*>(dQ), reinterpret_cast<const nv_bfloat16*>(dK), 
        reinterpret_cast<const nv_bfloat16*>(dV), dO, seqLength, 1);

    cudaMemcpy(bf16_hO, dO, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++)
    {
        hO[i] = __bfloat162float(bf16_hO[i]);
    }
    /*
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
    */
    printf("\n");
}
