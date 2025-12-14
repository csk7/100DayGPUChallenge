#include <torch/extension.h>
#include <cuda_bf16.h>

using AttentionFn = void(const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V, const nv_bfloat16* O, int seqLength, int batchSize);

AttentionFn flashAttention2_v1;

template<AttentionFn attention>
at::Tensor sdpa(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V)
{
    const int batchSize = Q.size(0);
    const int seqLength = Q.size(1);

    at:Tensor O = at::empty_like(Q);

    auto Q_ptr = reinterpret_Cast<const nv_bfloat16*>(Q.data_ptr());
    auto K_ptr = reinterpret_Cast<const nv_bfloat16*>(K.data_ptr());
    auto V_ptr = reinterpret_Cast<const nv_bfloat16*>(V.data_ptr());
    auto O_ptr = reinterpret_Cast<nv_bfloat16*>(O.data_ptr());

    attention(Q_ptr, K_ptr, V_ptr, O_ptr, seqLength, batchSize);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("flashAttention2_v1", torch::wrap_pybind_function(&flashAttention2_v1));
}