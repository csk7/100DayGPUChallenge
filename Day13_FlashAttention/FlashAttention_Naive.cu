#include<iostream>
#include<cuda.h>
#include<cmath>
#include<random>
#include<cassert>

using namespace std;

#include "flashAttentCpu.cuh"
#include "FlashAttention_utils.cuh"

#define TX_PER_BLOCK 16

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

void setVal2d(float** inpArr, float val, int N, int d)
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<d; j++)
        {
            inpArr[i][j] = val;
        }
    }
}

void setVal1d(float* inpArr, float val, int N)
{
    for(int i=0; i<N; i++)
    {
        inpArr[i] = val;
    }
}
template<const int Br=32, const int Bc = 32 , const int TM = 1, const int TN = 1, const int d = 32, const int WARPSIZE>
__global__ void flashAttentionKernel(float* Q, float* K, float* V, float* O, float* m ,float* l, int N)
{
    __shared__ float shQ[Br*d];
    __shared__ float shK_T[Bc*d];
    __shared__ float shV[Bc*d];
    __shared__ float shO[Br*d];
    __shared__ float shM_i[Br];
    __shared__ float shL_i[Br];

    __shared__ float shS_P[Br*Bc];
    __shared__ float shM_ij[Br];
    __shared__ float shL_ij[Br];
    __shared__ float shM_New_i[Br];
    __shared__ float shL_New_i[Br];

    const int blockRow = blockIdx.x*Br;
    if(blockRow > N) return; //Exit condition
    

    Q += blockRow*d;
    O += blockRow*d;
    m += blockRow;
    l += blockRow;
    
    load_L_i_M_i<Br>(shM_i, shL_i, m, l);
    
    load_O<Br>(O, shO, d);
    load_Q<Br>(Q, shQ, N, d);

    for(int j=0; j<N; j+=Bc) //j is from FA paper
    {
        load_K_V<Bc>(K, V, shK_T, shV, N, d);
        __syncthreads();
        matMulS<Br, Bc, TM, TN>(shQ, shK_T, shS_P, d);
        __syncthreads();
        rowMax_calculateP_rowSum<Br, Bc, TM, WARPSIZE>(shS_P, shM_ij, shL_ij);
        __syncthreads();
        calculate_Mnew_i_Lnew_i<Br>(shM_i, shL_i, shM_ij, shM_New_i, shL_New_i, shL_ij);
        __syncthreads();
        matMulPV_Update_O<Br,Bc, d, TM, TN>(shV, shS_P, shO, shM_ij, shM_New_i, shM_i, shL_New_i, shL_i, N, d);
        __syncthreads();
        copy_L_i_M_i<Br>(shM_i, shL_i, shM_New_i, shL_New_i);

        K += Bc*d;
        V += Bc*d;
    }
    write_O(O, shO, d);

}

template<const int Br=32, const int Bc = 32 , const int TM = 1, const int TN = 1, const int d = 32, const int WARPSIZE>
void gpuFlashAttention(float** h_Q, float** h_K, float** h_V, float** h_O, int N) //N is sequence length. d is the head dim.
{
    float* h_Q_1D = convert_2D_to_1D(h_Q, N, d);
    float* h_K_1D = convert_2D_to_1D(h_K, N, d);
    float* h_V_1D = convert_2D_to_1D(h_V, N, d);
    setVal2d(h_O, 0.0f, N, d);
    
    float* l = new float[N];
    float* m = new float[N];

    setVal1d(l, 0.0f, N);
    setVal1d(m, -INFINITY, N);

    //Declare device variables
    float* Q;
    float* K;
    float* V;
    float* O;
    float* d_l;
    float* d_m;

    int size = N * d * sizeof(float);
    int sizeL_M = N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&Q, size));
    CUDA_CHECK(cudaMalloc((void**)&K, size));
    CUDA_CHECK(cudaMalloc((void**)&V, size));
    CUDA_CHECK(cudaMalloc((void**)&O, size));
    CUDA_CHECK(cudaMalloc((void**)&d_l, sizeL_M));
    CUDA_CHECK(cudaMalloc((void**)&d_m, sizeL_M));

    //Copy values to device
    cudaMemcpy(Q, h_Q_1D, size, cudaMemcpyHostToDevice);
    cudaMemcpy(K, h_K_1D, size, cudaMemcpyHostToDevice);
    cudaMemcpy(V, h_V_1D, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, sizeL_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, sizeL_M, cudaMemcpyHostToDevice);

    //Call Kernel
    dim3 blockSize(TX_PER_BLOCK,1,1);

    dim3 gridSize(CEIL_CUSTOM(N,Br),1, 1);
    //size_t shMemSize = (2*BR*d + 2*BC*d)*sizeof(float);

    flashAttentionKernel<Br, Bc, TM, TN, d, WARPSIZE><<<gridSize, blockSize>>>(Q, K, V, O, d_m, d_l, N);
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

int main()
{
    int N = 8;
    const int d = 4;
    const int WARPSIZE = 4;

    float** h_Q;
    float** h_K;
    float** h_V;
    float** cpu_h_O;
    float** gpu_h_O;

    h_Q = assignHostSpace(N, d);
    h_K = assignHostSpace(N, d);
    h_V = assignHostSpace(N, d);
    cpu_h_O = assignHostSpace(N, d);
    gpu_h_O = assignHostSpace(N, d);

    assignHostValues(h_Q, N, d);
    assignHostValues(h_K, N, d);
    assignHostValues(h_V, N, d);

    attentionCpu(h_Q, h_K, h_V, cpu_h_O, N, d);
    gpuFlashAttention<4, 4, 1, 1, d, WARPSIZE>(h_Q, h_K, h_V, gpu_h_O, N);
    
    mismatch2D(cpu_h_O, gpu_h_O, N, d);

    return;

    
}