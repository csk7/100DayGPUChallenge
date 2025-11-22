#include<iostream>
#include<cuda.h>

#define H_TILE_WIDTH 16
#define H_BLOCK_TILE_COL_STRIDE 4
#define H_BLOCK_TILE_ROW_STRIDE 4

#define CEIL_CUSTOM(M, N) (((M) + (N) - 1)/(N))

float** assignHostSpace(int rows, int cols)
{
    float** hostArr;
    hostArr = new float*[rows];
    for(int i=0; i<rows; i++)
    {
        hostArr[i] = new float[cols];
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = 0;
        }
    }
    return hostArr;
}

void assignHostValues(float** hostArr, int rows, int cols)
{
    for(int i=0;i<rows;i++)
    {
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = float(uint(i*cols + j)%100);
        }
    }
}

void matMulCpu(float** h_A, float** h_B, float** h_C, int M, int N, int K)
{
    float pSum = 0.0;
    for(int i = 0; i<M; i++)
    {
        for(int j=0; j<K; j++)
        {
            pSum = 0.0;
            for(int inLoop = 0; inLoop<N; inLoop++)
            {
                pSum+= (h_A[i][inLoop]*h_B[inLoop][j]);
            }
            h_C[i][j] = pSum;
        }
    }
}

void mismatch2D(float** cpuArr, float** gpuArr, int row, int col)
{
    int flag = 1;
    for(int i=0; i<row; i++)
    {
        for(int j = 0; j<col; j++)
        {
            if(cpuArr[i][j] != gpuArr[i][j])
            {
                flag = 0;
                printf("Mismatch at : (%d,%d), CPU: %f ; GPU: %f \n",i,j,cpuArr[i][j],gpuArr[i][j]);
            }
        }
    }
    if(flag == 1)
        printf("Success \n");
}

float* convert_2D_to_1D(float** inpArr, int rows, int cols)
{
    float* outArr;
    outArr = new float[rows*cols];
    for(int i = 0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            outArr[i*cols + j] = inpArr[i][j];
        }
    }
    return outArr;
}

void convert_1D_to_2D(float* inpArr, float** outArr, int rows, int cols)
{
    for(int i=0 ;i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            outArr[i][j] = inpArr[i*cols + j];
        }
    }
}

template<const int TILE_WIDTH, const int BLOCK_TILE_COL_STRIDE, const int BLOCK_TILE_ROW_STRIDE>
__global__ void matMulKernel(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    /* Block Tiling. This allows data reuse of A. So the AI increase as you can do more compute for same A block that is read into shared Mem.
    Cols in B are accessed sequentially.
    */
    int row = blockIdx.y * (BLOCK_TILE_ROW_STRIDE * blockDim.y) + threadIdx.y;
    int col = blockIdx.x * (BLOCK_TILE_COL_STRIDE * blockDim.x) + threadIdx.x; //Effective BlockSize in Col Dimension. This is also reflected in the grid Dimension declaration.

    extern __shared__ float shMem[];
    
    float* MdS = shMem; //MdS contains the starting address
    float* NdS = &shMem[BLOCK_TILE_ROW_STRIDE*TILE_WIDTH*TILE_WIDTH];

    float pSum[BLOCK_TILE_ROW_STRIDE * BLOCK_TILE_COL_STRIDE] = {0.0}; //Multiple pSum local variables

    for(int ph = 0; ph<((N+TILE_WIDTH-1)/TILE_WIDTH); ph++)
    {
        //Load shared Mem
        for(int idx_Block1D = 0; idx_Block1D<(BLOCK_TILE_ROW_STRIDE*TILE_WIDTH); idx_Block1D+=TILE_WIDTH)
        {
            if((row + idx_Block1D)<M && (ph*TILE_WIDTH + threadIdx.x)<N)
                MdS[(threadIdx.y + idx_Block1D)*TILE_WIDTH + threadIdx.x] = d_A[(row + idx_Block1D)*N + (ph*TILE_WIDTH + threadIdx.x)];
            else
                MdS[(threadIdx.y + idx_Block1D)*TILE_WIDTH + threadIdx.x] = 0.0;
        }

        for(int idx_Block1D = 0; idx_Block1D<(BLOCK_TILE_COL_STRIDE*TILE_WIDTH); idx_Block1D+=TILE_WIDTH)
        {
            if((ph*TILE_WIDTH + threadIdx.y)<N && (col + idx_Block1D)<K)
            {
                //Each thread has to load multiple NdS from d_B. Jump for same thread is TILE_WIDTH
                //The NdS no.of cols also increase --> y index jump change
                NdS[threadIdx.y*(BLOCK_TILE_COL_STRIDE*TILE_WIDTH) + idx_Block1D + threadIdx.x] = d_B[(ph*TILE_WIDTH + threadIdx.y)*K + col + idx_Block1D];
            }
            else
            {
                NdS[threadIdx.y*(BLOCK_TILE_COL_STRIDE*TILE_WIDTH) + idx_Block1D + threadIdx.x] = 0.0;
            }
        }
        __syncthreads();
        //Partial dot product
        float tempMdS[BLOCK_TILE_ROW_STRIDE] = {0.0};
        float tempNdS[BLOCK_TILE_COL_STRIDE] = {0.0};
        for(int i = 0; i<TILE_WIDTH; i++)
        {
            for(int idx_BlockRow = 0; idx_BlockRow < BLOCK_TILE_ROW_STRIDE; idx_BlockRow++)
            {
                tempMdS[idx_BlockRow] = MdS[(threadIdx.y+idx_BlockRow*TILE_WIDTH)*TILE_WIDTH + i]; //Cache MdS in temp local reg. 
            }
            for(int idx_BlockCol = 0; idx_BlockCol<BLOCK_TILE_COL_STRIDE; idx_BlockCol++)
            {
                tempNdS[idx_BlockCol] = NdS[i*(BLOCK_TILE_COL_STRIDE*TILE_WIDTH) + idx_BlockCol*TILE_WIDTH + threadIdx.x];
            }
            for(int idx_BlockRow = 0; idx_BlockRow < BLOCK_TILE_ROW_STRIDE; idx_BlockRow++)
            {
                for(int idx_BlockCol = 0; idx_BlockCol<BLOCK_TILE_COL_STRIDE; idx_BlockCol++)
                {
                    pSum[idx_BlockRow*BLOCK_TILE_COL_STRIDE + idx_BlockCol] += (tempMdS[idx_BlockRow]* tempNdS[idx_BlockCol]); //Row jump is scaled, TILE_WIDTH Col increase for evry iter of inner loop
                }
            }
        }
        __syncthreads();
    }
    for(int idx_BlockRow = 0; idx_BlockRow < BLOCK_TILE_ROW_STRIDE; idx_BlockRow++)
    {
        for(int idx_BlockCol = 0; idx_BlockCol<BLOCK_TILE_COL_STRIDE; idx_BlockCol++)
        {
            if((row+idx_BlockRow*TILE_WIDTH)<M && (col + (idx_BlockCol*TILE_WIDTH))<K)
            {
                d_C[(row+idx_BlockRow*TILE_WIDTH)*K + (idx_BlockCol*TILE_WIDTH) + col] = pSum[idx_BlockRow*BLOCK_TILE_COL_STRIDE + idx_BlockCol]; 
            }
        }
    }
}

void matMulGpu(float** h_A, float** h_B, float** h_C, int M, int N, int K)
{
    //Prep Host values
    float* h_A_1D = convert_2D_to_1D(h_A, M, N);
    float* h_B_1D = convert_2D_to_1D(h_B, N, K);

    //Declare device variables
    float* d_A;
    float* d_B;
    float* d_C;
    int sizeA = M * N * sizeof(float);
    int sizeB = N * K * sizeof(float);
    int sizeC = M * K * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    //Copy values to device
    cudaMemcpy(d_A, h_A_1D, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_1D, sizeB, cudaMemcpyHostToDevice);

    //Call Kernel
    dim3 blockSize(H_TILE_WIDTH,H_TILE_WIDTH,1);
    dim3 gridSize(CEIL_CUSTOM(K,blockSize.x * H_BLOCK_TILE_COL_STRIDE),CEIL_CUSTOM(M,blockSize.y* H_BLOCK_TILE_ROW_STRIDE),1);
    size_t shMemSize = (H_BLOCK_TILE_ROW_STRIDE+H_BLOCK_TILE_COL_STRIDE)*H_TILE_WIDTH*H_TILE_WIDTH*sizeof(float);

    matMulKernel<H_TILE_WIDTH, H_BLOCK_TILE_COL_STRIDE, H_BLOCK_TILE_ROW_STRIDE><<<gridSize, blockSize, shMemSize>>>(d_A, d_B, d_C, M, N, K);

    //Copy values back
    float* h_C_1D = new float[M*K];
    cudaMemcpy(h_C_1D, d_C, sizeC, cudaMemcpyDeviceToHost);
    convert_1D_to_2D(h_C_1D, h_C, M, K);
    
    //Free device space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    //Declare host variables
    int M = 8162;
    int N = 6144;
    int K = 4092;

    float** h_A = assignHostSpace(M, N);
    float** h_B = assignHostSpace(N, K);
    float** h_C_cpu = assignHostSpace(M, K);
    float** h_C_gpu = assignHostSpace(M, K);

    //Assign values
    assignHostValues(h_A, M, N);
    assignHostValues(h_B, N, K);

    //Call CPU and GPU
    //matMulCpu(h_A, h_B, h_C_cpu, M, N, K);
    matMulGpu(h_A, h_B, h_C_gpu, M, N, K);

    //compare
    //mismatch2D(h_C_cpu, h_C_gpu, M, K);

    return 0;
}
