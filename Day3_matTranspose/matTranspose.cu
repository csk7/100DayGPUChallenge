_global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float shMem[TILE_WIDTH][TILE_WIDTH+1];

    if(row<rows && col<cols)
    {
        shMem[threadIdx.x][threadIdx.y] = input[row*cols + col];
    }
    else
        shMem[threadIdx.x][threadIdx.y] = 0.0 ;

    __syncthreads();
    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;

    if(row<cols && col<rows)
        output[row * rows + col] = shMem[threadIdx.y][threadIdx.x];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}