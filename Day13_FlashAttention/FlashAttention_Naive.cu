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

void setVal2d(float* inpArr, float val, int N, int d)
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
template<const int Br=32, const int Bc = 32 , const int TM = 1, const int TN = 1>
__global__ void flashAttentionKernel(float* Q, float* K, float* V, float* m ,float* l, int N, int d)
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
    __shared__ float shM_New_i[Bc];

    const int blockRow = blockIdx.x*Br;
    if(blockRow > N) return; //Exit condition

    Q += blockRow*d;
    O += blockRow*d;
    m += blockRow;
    l += blockRow;

    load_L_i_M_i(shM_i, shL_i, m, l);
    load_O(O, shO, d);

    for(int j=0; j<N; j+=Bc) //j is from FA paper
    {
        load_K_V<Br,Bc>(K, V, shK_T, shV);
        __syncthreads();
        loadQ_matMulS(Q, shQ, shS_P, d);
        __syncthreads();
        rowMax_calculateP_rowSum(shS_P, shM_ij, shL_ij);
        __syncthreads();
        calculate_Mnew_i_Lnew_i(shM_i, shL_i, shM_ij, shM_New_i, shL_New_i, shL_ij);
        __syncthreads();
        matMulPV_Update_O(shV, shS_P, shO, shM_ij, shM_New_i, shM_i, shL_New_i, shL_i, N, d);
        __synthreads();
        copy_L_i_M_i(shM_i, shL_i, shM_New_i, shL_New_i);

        K += Bc*d;
        V += Bc*d;
    }
    write_O(0, sh0, d);

}

void gpuFlashAttention(float** h_Q, float** h_K, float** h_V, float** h_O, int N, int d) //N is sequence length. d is the head dim.
{
    //Prep Host values
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

int main()
{
    const int N = 32;
    const int d = 32;
    
}