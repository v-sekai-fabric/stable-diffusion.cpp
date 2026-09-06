#!/usr/bin/env python3
"""Convert OmniGen2 (VectorSpaceLab/OmniGen2) safetensors → GGUF.

Milestone 1: emits a GGUF with the tensor names sd.cpp's model_loader looks
for. The header records the arch config so future milestones can validate.
Actual model_load acceptance in sd.cpp lands with Milestone 2 (block forward
+ VERSION_OMNIGEN2 dispatch in model_loader.cpp).

Usage:
    python convert_omnigen2_to_gguf.py \\
        --src /path/to/omnigen2 \\
        --dst /path/to/omnigen2.gguf \\
        --ftype f16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import gguf                                                                    # from `pip install gguf`
import numpy as np
from safetensors import safe_open


# Mapping from OmniGen2 diffusers tensor names → sd.cpp expected names.
# Structure mirrors src/model/diffusion/omnigen2.hpp block layout.
def name_map(src: str) -> str:
    m = src
    # Diffusers uses `transformer.` prefix on safetensors; sd.cpp expects
    # `model.diffusion_model.` (added later by model_loader). Strip here.
    for pfx in ("transformer.", "model.diffusion_model.transformer.", "model.diffusion_model."):
        if m.startswith(pfx):
            m = m[len(pfx):]
            break
    # Diffusers Attention names → our block layout (already matches, but
    # verify to_out.0 shape). No rename needed for the common cases.
    # feed_forward.linear_{1,2,3}: already match.
    # norm1, ffn_norm1, norm2, ffn_norm2: already match.
    # time_caption_embed.timestep_embedder.{linear_1,linear_2}: already match.
    # time_caption_embed.caption_embedder.0/1: already match after Python
    # Sequential -> .0/.1 index expansion (diffusers does this on save).
    return m


def convert(src: Path, dst: Path, ftype: str) -> None:
    if not src.exists():
        sys.exit(f"source directory not found: {src}")

    config_path = src / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found under {src}")
    config = json.loads(config_path.read_text())

    writer = gguf.GGUFWriter(str(dst), arch="omnigen2")
    # Record arch config in the GGUF header so future model_loader can pick it
    # up without hunting through weight shapes.
    writer.add_string("omnigen2.arch_source", "VectorSpaceLab/OmniGen2")
    for k in ("patch_size", "in_channels", "out_channels", "hidden_size",
              "num_layers", "num_refiner_layers", "num_attention_heads",
              "num_kv_heads", "multiple_of", "norm_eps", "text_feat_dim",
              "timestep_scale", "theta"):
        if k in config:
            v = config[k]
            if isinstance(v, bool):
                writer.add_bool(f"omnigen2.{k}", v)
            elif isinstance(v, int):
                writer.add_int32(f"omnigen2.{k}", v)
            elif isinstance(v, float):
                writer.add_float32(f"omnigen2.{k}", v)
    if "axes_dim_rope" in config:
        writer.add_array("omnigen2.axes_dim_rope", list(config["axes_dim_rope"]))
    if "axes_lens" in config:
        writer.add_array("omnigen2.axes_lens", list(config["axes_lens"]))

    # Find safetensors files (diffusers shards). Prefer index if present.
    index_path = src / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map: Dict[str, str] = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = [p.name for p in src.glob("*.safetensors")]
        weight_map = {}

    dtype_np = {"f16": np.float16, "f32": np.float32, "bf16": np.uint16}.get(ftype)
    if dtype_np is None:
        sys.exit(f"unknown ftype: {ftype} (want f16 / f32 / bf16)")

    total = 0
    for shard in shard_files:
        shard_path = src / shard
        if not shard_path.exists():
            print(f"skip missing shard: {shard}", file=sys.stderr)
            continue
        with safe_open(str(shard_path), framework="pt") as f:
            for src_name in f.keys():
                dst_name = name_map(src_name)
                tensor = f.get_tensor(src_name).contiguous()
                # bf16 stays as uint16 view; f16 / f32 use numpy directly.
                if ftype == "bf16":
                    array = tensor.view(np.uint16).numpy() if hasattr(tensor, "view") else tensor.numpy()
                else:
                    array = tensor.to(dtype=None).numpy().astype(dtype_np)
                writer.add_tensor(dst_name, array)
                total += 1
                if total % 50 == 0:
                    print(f"  wrote {total} tensors...", file=sys.stderr)

    print(f"total tensors written: {total}", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert OmniGen2 safetensors → GGUF")
    ap.add_argument("--src", type=Path, required=True, help="OmniGen2 model directory (containing config.json + *.safetensors)")
    ap.add_argument("--dst", type=Path, required=True, help="Output .gguf path")
    ap.add_argument("--ftype", default="f16", choices=("f16", "f32", "bf16"), help="Output tensor dtype (default f16)")
    args = ap.parse_args()
    convert(args.src, args.dst, args.ftype)


if __name__ == "__main__":
    main()
