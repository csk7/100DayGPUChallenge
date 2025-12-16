#include <torch/extension.h>
#include <cuda_bf16.h>
#include <iostream>

using AttentionFn = void(const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V, nv_bfloat16* O, int seqLength, int batchSize);

AttentionFn flashAttention2_v1;

template<AttentionFn attention>
at::Tensor sdpa(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V)
{
    const int batchSize = Q.size(0);
    const int seqLength = Q.size(2);

    at::Tensor O = at::zeros_like(Q);

    auto Q_ptr = reinterpret_cast<const nv_bfloat16*>(Q.data_ptr());
    auto K_ptr = reinterpret_cast<const nv_bfloat16*>(K.data_ptr());
    auto V_ptr = reinterpret_cast<const nv_bfloat16*>(V.data_ptr());
    auto O_ptr = reinterpret_cast<nv_bfloat16*>(O.data_ptr());

    printf("Hello CPP \n");

    attention(Q_ptr, K_ptr, V_ptr, O_ptr, seqLength, batchSize);


    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("sdpa_v1", torch::wrap_pybind_function(&sdpa<flashAttention2_v1>));
}