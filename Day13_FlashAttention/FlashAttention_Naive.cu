#include<cuda.h>
#include<iostream>

#defint TX_PER_BLOCK 256

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

template<const int Br=2, const int Bc = 2, , const int TM = 1, const int TN = 1>
__global__ void flashAttentionKernel(float* Q, float* K, float* V, int N, int d)
{
    __shared__ float shQ[Br*d];
    __shared__ float shK[Bc*d];
    __shared__ float shV[Bc*d];
    __shared__ float shO[Br*d];

    const int blockRow = blockIdx.x*Br;
    if(blockRow > N) return; //Exit condition

    Q += blockRow*d;
    O += blockRow*d;

    const int idxRow = threadIdx.x/(Bc/TN);
    const int idxCol = threadIdx.x%/(Bc/TN);

    for(int j=0; j<N; j+=Bc) //j is from FA paper
    {
        
    }

}

void gpuFlashAttention(float** h_Q, float** h_K, float** h_V, float** h_O, int N, int d) #N is sequence length. d is the head dim.
{
    //Prep Host values
    float* h_Q_1D = convert_2D_to_1D(h_Q, N, d);
    float* h_K_1D = convert_2D_to_1D(h_K, N, d);
    float* h_V_1D = convert_2D_to_1D(h_V, N, d);

    //Declare device variables
    float* Q;
    float* K;
    float* V;
    float* O;
    int size = N * d * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&Q, size));
    CUDA_CHECK(cudaMalloc((void**)&K, size));
    CUDA_CHECK(cudaMalloc((void**)&V, size));
    CUDA_CHECK(cudaMalloc((void**)&O, size));

    //Copy values to device
    cudaMemcpy(Q, h_Q_1D, size, cudaMemcpyHostToDevice);
    cudaMemcpy(K, h_K_1D, size, cudaMemcpyHostToDevice);
    cudaMemcpy(V, h_V_1D, size, cudaMemcpyHostToDevice);

    //Call Kernel
    dim3 blockSize(TX_PER_BLOCK,1,1);

    dim3 gridSize(CEIL_CUSTOM(N,BR),1, 1);
    size_t shMemSize = (2*Br*d + 2*Bc*d)*sizeof(float);

    matMulKernel<<<gridSize, blockSize, shMemSize>>>(Q, K, V, N, d);
    //Copy values back
    float* h_O_1D = new float[N*d];
    cudaMemcpy(h_O_1D, O, size, cudaMemcpyDeviceToHost);
    convert_1D_to_2D(h_O_1D, h_O, N, d);
    
    //Free device space
    cudaFree(Q);
    cudaFree(V);
    cudaFree(K);
    cudaFree(O);
}