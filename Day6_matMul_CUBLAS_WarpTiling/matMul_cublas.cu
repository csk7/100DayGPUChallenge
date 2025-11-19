#include <cublas_v2.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)  \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA error\n"; exit(1); \
    }

#define CHECK_CUBLAS(call)  \
    if((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error\n"; exit(1); \
    }


int main() {
    int M = 8162, K = 4096, N = 6144;

    std::vector<float> h_A(M*K);
    std::vector<float> h_B(K*N);
    std::vector<float> h_C(M*N, 0.0f);

    // Fill with some values
    for(int i=0;i<M;i++)
    {
        for(int j=0; j<K; j++)
        {
            h_A[i*K + j] = float(uint(i*K + j)%100);
        }
    }

    for(int i=0;i<K;i++)
    {
        for(int j=0; j<N; j++)
        {
            h_B[i*N + j] = float(uint(i*N + j)%100);
        }
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // row-major → swap A/B + transpose both
    CHECK_CUBLAS(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,      // transA
            CUBLAS_OP_N,      // transB
            N,                // m
            M,                // n
            K,                // k
            &alpha,
            d_B, N,           // A is MxK (row-major)
            d_A, K,           // B is KxN (row-major)
            &beta,
            d_C, N            // C is MxN
        )
    );

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}
