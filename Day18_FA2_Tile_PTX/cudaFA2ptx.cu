#include<iostream>
#include<cassert>
#include<cuda.h>
#include<cuda_bf16.h>

#include"common.cuh"

/* Inputs are Q-> B, L, d and KV-> B, L, d and output is B, L, d*/

template<const int batchSize = 1, const int Br=128, const int Bc=128, const int d=128, const int numThreads = 128, const float scaling_factor = 0.088> 
__global__ void __launch_bounds__(numThreads) flashAttention2(const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V, nv_bfloat16* O, const int seqLength)
{
    const int row = blockIdx.x*Br + blockIdx.y*seqLength;
    if((blockIdx.x>batchSize) or ((blockIdx.z*Br)>seqLength))
    {
        return;
    }
    //Block pointer moved
    Q += row*d;
    K += (blockDim.y*seqLength)*d;
    V += (blockDim.y*seqLength)*d;
    O += row*d;

    //Shared Mem declaration
    extern __shared__ nv_bfloat16 shMem[];
    const uint32_t QshMem = __cvta_generic_to_shared(shMem);
    const uint32_t KshMem = QshMem;

    //Thread Index to Warp Id and lane Id
    const int tid = threadIdx.x;
    const int warpId = tid/WARPSIZE;
    const int laneId = tid%WARPSIZE;
    const int numWarps = blockDim.x/WARPSIZE; 
    const int blockQperWarp = Br/numWarps;

    //Tiled MMA size
    const int MMA_M = 16;
    const int MMA_K = 16;
    const int MMA_N = 8;

    //Reg declaration
    uint32_t Qreg[blockQperWarp/MMA_M][d/MMA_K][4]; //4  16*16/32 following below logic (2  bf16 per reg)
    uint32_t Kreg[Bc/MMA_N][d/MMA_K][2];
    float Sreg[blockQperWarp/MMA_M][Bc/MMA_N][4]; //fp32. So per warp total space/32 --> 16*8/32
    float Oreg[blockQperWarp/MMA_M][d/MMA_K][4] = {}; //4  is number of tiles per warp

    float rowMax[blockQperWarp/MMA_M][2]; 
    for(int i=0; i<(blockQperWarp/MMA_M); i++)
        rowMax[i] = {-INFINITY, -INFINITY};

    float rowSumExp[blockQperWarp/MMA_M][2];
    //Global to shMem transfer
    //Load Q (Br*d)
    const int numElementsPerLoad = 8;
    assert((numElementsPerLoad*2) == 4 or (numElementsPerLoad*2) == 8 or (numElementsPerLoad*2) == 16);
    assert(numThreads >= (d/numElementsPerLoad));

    globalToShared<Br, d>(QshMem, Q, numElementsPerLoad);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    //Shared to Reg
    for(int mma_id_row=0; mma_id_row<blockQperWarp; mma_id_row+=MMA_M)
    {
        for(int mma_id_col=0; mma_id_col<d; mma_id_col+=MMA_K)
        {
            const int idxRow = warpId*blockQperWarp + mma_id_row + laneId%16; //16 is a function of how many tiles (rowWise) are there in the m8n8 load matrix. 2 tiles(2*m8) so 16
            const int idxCol = mma_id_col + laneId/16*8; //16 --> from above, ldMatrix m8n8. n8 contributes to this;
            const uint32_t srcShAddress = QshMem + (idxRow*d + idxCol)*sizeof(nv_bfloat16);
            sharedToRegx4(Qreg[mma_id_row/MMA_M][mma_id_col/MMA_K], srcShAddress);
        }
    }

    for(int idx_KV = 0; idx_KV<L; idx_KV+=Bc)
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

        //MMA
        for(int i=0; i<(blockQperWarp/MMA_M), i++)
        {
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                for(int k=0; k<(d/MMA_K); k++)
                {
                    mma_m16n8k16(Sreg[i][j], Qreg[i][k], Kreg[j][k]);
                }
            }
        }

        //Head scaling and Row max
        for(int i=0; i<(blockQperWarp/MMA_M), i++)
        {
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                for(int k=0; k<4; k++)
                {
                    Sreg[i][j][k] *= scaling_factor;
                }
            }
            //row max, online softmax
            float* maxTileRow = {-INFINTIY, -INFINITY};
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                float* tempReg = Sreg[i][j];
                maxTileRow[0] = max(maxTileRow[0], max(tempReg[0], tempReg[1]));
                maxTileRow[1] = max(maxTileRow[1], max(tempReg[2], tempReg[3]));
            } //We close here as we get all the values pertianing to this row. No need separately. Clever

            maxTileRow[0] = max(maxTileRow[0], __shfl_xor_sync('OxFFFFFFFF', maxTileRow[0], 1));
            maxTileRow[0] = max(maxTileRow[0], __shfl_xor_sync('OxFFFFFFFF', maxTileRow[0], 2));
            maxTileRow[1] = max(maxTileRow[1], __shfl_xor_sync('OxFFFFFFFF', maxTileRow[1], 1));
            maxTileRow[1] = max(maxTileRow[1], __shfl_xor_sync('OxFFFFFFFF', maxTileRow[1], 2));
            

            maxTileRow[0] = max(maxTileRow[0], rowMax[i][0]);
            maxTileRow[1] = max(maxTileRow[1], rowMax[i][1]);

            //rescale
            float rescale[2];
            rescale[0] = expf(rowMax[i][0] - maxTileRow[0]);
            rescale[1] = expf(rowMax[i][1] - maxTileRow[1]);

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
            float* sumexp = {0.0, 0.0};
            for(int j=0; j<(Bc/MMA_N); j++)
            {
                S[i][j][0] = expf(S[i][j][0] - maxTileRow[0]);
                S[i][j][1] = expf(S[i][j][1] - maxTileRow[0]);
                S[i][j][2] = expf(S[i][j][0] - maxTileRow[1]);
                S[i][j][3] = expf(S[i][j][1] - maxTileRow[1]);

                sumexp[0] += S[i][j][0] + S[i][j][1];
                sumexp[1] += S[i][j][2] + S[i][j][3];
            }

            sumexp[0] += __shfl_xor_sync('0xFFFFFFFF', sumexp[0], 1);
            sumexp[0] += __shfl_xor_sync('0xFFFFFFFF', sumexp[0], 2);
            sumexp[1] += __shfl_xor_sync('0xFFFFFFFF', sumexp[1], 1);
            sumexp[1] += __shfl_xor_sync('0xFFFFFFFF', sumexp[1], 2);
            

            rowSumExp[i][0] = rowSumExp[i][0]*rescale[0] +  sumexp[0];
            rowSumExp[i][1] = rowSumExp[i][1]*rescale[1] + sumexp[1];


            

        }



        K += Bc*d;
        V += Bc*d;
    }

    for(int mma_id_row=0; mma_id_row<(blockQperWarp/MMA_M); mma_id_row++)
    {
        for(int mma_id_col=0; mma_id_col<(d/MMA_N); mma_id_col++)
        {
            const int idxRow = warpId*blockQperWarp + mma_id_row + laneId/4; //So, 32 tx write a m16n8 output. So each tx writes 4 outs. fp32 out each occupy 1 reg.
            //Tx 0 gets 4 outputs. Row0, Col0, Col1 stored in Reg0 and Reg1 as fp32. If it was bf16 then Both in reg0. 
            //Only half the ouput regs are written by a tx in one row. Other half to m16/2 row. In fp32 you need 4 tx to fill a row. 
            //Each tx rows are current row + m16/2.   
            const int idxCol = mma_id_col + laneId%4*2; //2 as each row is writing 2 elems. 8 n8 so %4.  
            float* temp = Oreg[mma_id_row][mma_id_col];
            reinterpret_cast<nv_bfloat162*>(&O[idxRow*d + idxCol])[0] = __float22bfloat162_rn({temp[0], temp[1]});
            reinterpret_cast<nv_bfloat162*>(&O[(idxRow + 8)*d + idxCol])[0] = __float22bfloat162_rn({temp[0], temp[1]});
        }
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