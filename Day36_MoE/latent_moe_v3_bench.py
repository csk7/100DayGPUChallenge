"""Benchmark all v3 MoE variants."""
import gc
import time
import torch
import torch._dynamo

# Allow torch.compile to trace through .item() calls (e.g. cnt_max in Original).
# The compiled graph will recompile if the scalar value changes across calls,
# but within a fixed-T benchmark run the value is stable.
torch._dynamo.config.capture_scalar_outputs = True


class Config:
    def __init__(self):
        self.hidden_dim = 2688
        self.n_experts = 128
        self.n_shared_experts = 1
        self.top_k = 6
        self.n_groups = 8
        self.topk_groups = 1
        self.norm_weights = True
        self.scaling_factor = 2.5
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"
        self.moe_intermediate_size = 1856
        self.moe_shared_expert_intermediate_size = 3712
        self.d = self.hidden_dim
        self.m = self.moe_intermediate_size


def bench_model(model_cls, cfg, x, dtype=torch.bfloat16, warmup=5, iters=20,
                seed=42, compiled=False):
    torch.manual_seed(seed)
    model = model_cls(cfg).to(device="cuda", dtype=dtype).eval()
    if compiled:
        model = torch.compile(model, mode="reduce-overhead")
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / iters) * 1000.0
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return avg_ms


def correctness_check(model_cls, ref_cls, cfg, x, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    ref = ref_cls(cfg).to(device="cuda", dtype=dtype).eval()
    torch.manual_seed(seed)
    test = model_cls(cfg).to(device="cuda", dtype=dtype).eval()
    test.load_state_dict(ref.state_dict(), strict=False)
    with torch.inference_mode():
        y_ref = ref(x).float().cpu()

    del ref
    torch.cuda.empty_cache()
    gc.collect()

    with torch.inference_mode():
        y_test = test(x).float().cpu()

    del test
    torch.cuda.empty_cache()
    gc.collect()

    diff = (y_ref - y_test).abs()
    return diff.max().item(), diff.mean().item()


def main():
    from latent_moe_v3_original import MoE_V3_Original
    from latent_moe_v3_kernel1 import MoE_V3_Kernel1
    from latent_moe_v3_kernel2 import MoE_V3_Kernel2
    from latent_moe_v3_kernel3 import MoE_V3_Kernel3

    cfg = Config()
    dtype = torch.bfloat16

    # (name, cls, compiled)
    variants = [
        ("Original",               MoE_V3_Original, False),
        #("Original [compiled]",    MoE_V3_Original, True),
        ("Kernel1 (GEMM+sqReLU)",  MoE_V3_Kernel1,  False),
        ("Kernel2 (+scatter)",     MoE_V3_Kernel2,   False),
        ("Kernel3 (+stream)",      MoE_V3_Kernel3,   False),
    ]

    for T in [32, 256, 1024]:
        B = 1
        g = torch.Generator(device="cuda").manual_seed(1234)
        x = torch.randn((B, T, cfg.hidden_dim), device="cuda", dtype=dtype, generator=g)
        print(f"\n{'='*65}")
        print(f"  T={T}  (B={B}, K={cfg.top_k}, N={cfg.n_experts})")
        print(f"{'='*65}")
        print(f"  {'Variant':<35}  {'Time':>8}  {'vs Orig':>8}")
        print(f"  {'-'*55}")

        ref_ms = None
        for name, cls, use_compile in variants:
            ms = bench_model(cls, cfg, x, dtype=dtype, compiled=use_compile)
            if ref_ms is None:
                ref_ms = ms
            speedup = ref_ms / ms
            print(f"  {name:<35}  {ms:>7.3f}ms  {speedup:>7.2f}x")

        if T <= 32:
            print(f"\n  Correctness vs Original:")
            torch.compiler.reset()
            torch.cuda.empty_cache()
            gc.collect()
            for name, cls, _ in variants[2:]:
                max_err, mean_err = correctness_check(cls, MoE_V3_Original, cfg, x, dtype=dtype)
                print(f"    {name:30s}  max={max_err:.4f}  mean={mean_err:.6f}")
        else:
            print(f"\n  (correctness check skipped at T={T} to avoid OOM)")


if __name__ == "__main__":
    main()
