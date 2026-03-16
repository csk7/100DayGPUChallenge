"""
Extract layer-7 (index 6) MoE weights from nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
and save them with filenames matching latent_moe.py's read_weights_from_file() convention.

Directory layout produced:
    /weights/nemotron_nano/
        router.pt                    -> {"weight": (128, 2688), "bias": (128,)}
        layer0_ffn_exp_{i}_0.pt      -> {"weight": (1856, 2688), "bias": (1856,)}   # up_proj   (i = 0..127)
        layer0_ffn_exp_{i}_1.pt      -> {"weight": (2688, 1856), "bias": (2688,)}   # down_proj (i = 0..127)
        layer0_ffn_sh_0_0.pt         -> {"weight": (3712, 2688), "bias": (3712,)}   # shared up_proj
        layer0_ffn_sh_0_1.pt         -> {"weight": (2688, 3712), "bias": (2688,)}   # shared down_proj

NOTE:
    Nemotron's expert FFN is NOT gated — there is no gate/third projection.
    The forward is:  down_proj(relu²(up_proj(x)))
    So there are NO layer0_ffn_exp_{i}_2 files.
    latent_moe.py must be updated to remove the gate branch in expert_ffn_fwd().
"""

import os
import json
import torch
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = Path(os.path.join(_BASE, "data", "nemotron_nano"))
WEIGHT_DIR = Path(os.path.join(_BASE, "weights", "nemotron_nano"))
LAYER_IDX = 6  # 0-based; 7th layer overall, which is 'E' (MoE) in the hybrid pattern


def download_model():
    """Download the model to MODEL_DIR if not already present."""
    from huggingface_hub import snapshot_download

    marker = MODEL_DIR / "config.json"
    if marker.exists():
        print(f"Model already present at {MODEL_DIR}")
        return

    print(f"Downloading {MODEL_ID} to {MODEL_DIR} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    print("Download complete.")


def verify_moe_layer():
    """Verify that LAYER_IDX points to an MoE ('E') layer."""
    config_path = MODEL_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    pattern = config["hybrid_override_pattern"]
    if LAYER_IDX >= len(pattern):
        raise ValueError(f"LAYER_IDX={LAYER_IDX} out of range (pattern length {len(pattern)})")
    if pattern[LAYER_IDX] != "E":
        raise ValueError(
            f"Layer {LAYER_IDX} is '{pattern[LAYER_IDX]}' (not MoE). "
            f"Pattern: {pattern}"
        )
    print(f"Layer {LAYER_IDX} confirmed as MoE ('E')")

    n_routed = config["n_routed_experts"]
    n_shared = config["n_shared_experts"]
    print(f"  routed experts: {n_routed}, shared experts: {n_shared}")
    print(f"  moe_intermediate_size: {config['moe_intermediate_size']}")
    print(f"  shared_expert_intermediate_size: {config['moe_shared_expert_intermediate_size']}")
    print(f"  hidden_size: {config['hidden_size']}")
    print(f"  mlp_bias: {config['mlp_bias']}")
    return config


def load_tensor(index_data, key):
    """Load a single tensor from the correct safetensors shard."""
    from safetensors import safe_open

    weight_map = index_data["weight_map"]
    if key not in weight_map:
        raise KeyError(f"Key '{key}' not found in weight map")

    shard_file = MODEL_DIR / weight_map[key]
    with safe_open(str(shard_file), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def extract_and_save(config):
    """Extract MoE weights for LAYER_IDX and save with latent_moe.py naming."""
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    index_path = MODEL_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index_data = json.load(f)

    prefix = f"backbone.layers.{LAYER_IDX}.mixer"
    hidden_size = config["hidden_size"]
    n_routed = config["n_routed_experts"]
    n_shared = config["n_shared_experts"]
    has_bias = config["mlp_bias"]

    # --- Router ---
    print("Extracting router weights ...")
    router_w = load_tensor(index_data, f"{prefix}.gate.weight")
    router_bias = torch.zeros(router_w.shape[0])
    torch.save({"weight": router_w, "bias": router_bias}, WEIGHT_DIR / "router.pt")
    print(f"  router.pt  weight={tuple(router_w.shape)}  bias={tuple(router_bias.shape)}")

    # --- Routed experts ---
    for i in range(n_routed):
        up_key = f"{prefix}.experts.{i}.up_proj.weight"
        dn_key = f"{prefix}.experts.{i}.down_proj.weight"

        up_w = load_tensor(index_data, up_key)
        dn_w = load_tensor(index_data, dn_key)

        up_bias = torch.zeros(up_w.shape[0]) if not has_bias else load_tensor(index_data, f"{prefix}.experts.{i}.up_proj.bias")
        dn_bias = torch.zeros(dn_w.shape[0]) if not has_bias else load_tensor(index_data, f"{prefix}.experts.{i}.down_proj.bias")

        torch.save({"weight": up_w, "bias": up_bias}, WEIGHT_DIR / f"layer0_ffn_exp_{i}_0.pt")
        torch.save({"weight": dn_w, "bias": dn_bias}, WEIGHT_DIR / f"layer0_ffn_exp_{i}_1.pt")

        if i % 16 == 0 or i == n_routed - 1:
            print(f"  expert {i:>3d}:  _0 (up)={tuple(up_w.shape)}  _1 (down)={tuple(dn_w.shape)}")

    # --- Shared experts ---
    for i in range(n_shared):
        up_key = f"{prefix}.shared_experts.up_proj.weight"
        dn_key = f"{prefix}.shared_experts.down_proj.weight"

        up_w = load_tensor(index_data, up_key)
        dn_w = load_tensor(index_data, dn_key)

        up_bias = torch.zeros(up_w.shape[0]) if not has_bias else load_tensor(index_data, f"{prefix}.shared_experts.up_proj.bias")
        dn_bias = torch.zeros(dn_w.shape[0]) if not has_bias else load_tensor(index_data, f"{prefix}.shared_experts.down_proj.bias")

        torch.save({"weight": up_w, "bias": up_bias}, WEIGHT_DIR / f"layer0_ffn_sh_{i}_0.pt")
        torch.save({"weight": dn_w, "bias": dn_bias}, WEIGHT_DIR / f"layer0_ffn_sh_{i}_1.pt")
        print(f"  shared {i}:  _0 (up)={tuple(up_w.shape)}  _1 (down)={tuple(dn_w.shape)}")

    print(f"\nAll weights saved to {WEIGHT_DIR}/")
    print(f"  Router:          1 file")
    print(f"  Routed experts:  {n_routed * 2} files (128 experts × 2 projections)")
    print(f"  Shared experts:  {n_shared * 2} files")
    print(f"  Total:           {1 + n_routed * 2 + n_shared * 2} .pt files")
    print()
    print("WARNING: Nemotron expert FFN has NO gate projection (no _2 files).")
    print("         latent_moe.py must use:  down_proj(relu²(up_proj(x)))")
    print("         Remove the gate branch (weight_name_2 / intermediate_gate) from expert_ffn_fwd().")


if __name__ == "__main__":
    download_model()
    config = verify_moe_layer()
    extract_and_save(config)
