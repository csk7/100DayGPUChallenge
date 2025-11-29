#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*Gemini generated*/
// Helper macro for checking CUDA error codes
#define CUDA_CHECK(call) \
{ \
    const cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Helper macro for checking cuBLAS error codes
#define CUBLAS_CHECK(call) \
{ \
    const cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) \
    { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
}

void sgemv_example(int M, int N) {
    cublasHandle_t handle;
    float *h_A, *h_x, *h_y, *d_A, *d_x, *d_y;
    float alpha = 1.0f; // Scaling factor for A*x
    float beta = 0.0f;  // Scaling factor for y (set to 0 for y = A*x)
    
    // Matrix A is M rows x N columns
    size_t size_A = (size_t)M * N * sizeof(float);
    size_t size_x = (size_t)N * sizeof(float);
    size_t size_y = (size_t)M * sizeof(float);

    // 1. Host Memory Allocation and Initialization
    h_A = (float*)malloc(size_A);
    h_x = (float*)malloc(size_x);
    h_y = (float*)malloc(size_y);
    
    // (Initialize h_A and h_x with your data here)

    // 2. cuBLAS Initialization
    CUBLAS_CHECK(cublasCreate(&handle));

    // 3. Device Memory Allocation
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_x));
    CUDA_CHECK(cudaMalloc((void**)&d_y, size_y));

    // 4. Copy data Host to Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));
    // Initialize d_y on device (or copy h_y if beta is non-zero)
    CUDA_CHECK(cudaMemset(d_y, 0, size_y)); 

    // 5. cuBLAS Sgemv Call (y = alpha*A*x + beta*y) 
    // Note: cuBLAS uses column-major order. If your matrix h_A is row-major (C-style), 
    // you typically transpose A using CUBLAS_OP_T and swap M and N in the call.
    // Assuming C-style (Row-Major) matrix A is used with Transpose operation:
    CUBLAS_CHECK(cublasSgemv(
        handle, 
        CUBLAS_OP_T, // Transpose: performs A^T * x (effectively A * x for row-major A)
        N,           // rows of A (M is the N parameter for A^T)
        M,           // columns of A (N is the M parameter for A^T)
        &alpha,      // pointer to alpha
        d_A,         // Device pointer to matrix A
        N,           // Leading dimension of A (lda) - distance between columns for A^T
        d_x,         // Device pointer to vector x
        1,           // Stride for x (incx)
        &beta,       // pointer to beta
        d_y,         // Device pointer to vector y
        1            // Stride for y (incy)
    ));

    // 6. Copy result Device to Host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost));

    // (Process result h_y here)

    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_A);
    free(h_x);
    free(h_y);
    CUBLAS_CHECK(cublasDestroy(handle));
}

int main() {
    sgemv_example(4096, 16*2048); // Example: 4x3 matrix * 3x1 vector = 4x1 vector
    return 0;
 }