# OmniGen2 architecture notes (sd.cpp port, milestone 1)

Source: `VectorSpaceLab/OmniGen2` (Apache-2.0), `omnigen2/models/transformers/{transformer_omnigen2.py, block_lumina2.py}`.

## Config (default from `OmniGen2Transformer2DModel.__init__`)

| field | default | notes |
|---|---|---|
| `patch_size` | 2 | 2×2 pixel-space patch |
| `in_channels` | 16 | VAE latent channels |
| `out_channels` | 16 (=in) | |
| `hidden_size` | 2304 | model width |
| `num_layers` | 26 | main transformer layers |
| `num_refiner_layers` | 2 | noise / ref_image / context refiners (three sets) |
| `num_attention_heads` | 24 | dim/head = 2304/24 = 96 |
| `num_kv_heads` | 8 | GQA (24 Q heads share 8 KV heads) |
| `multiple_of` | 256 | FFN inner-dim rounding |
| `ffn_dim_multiplier` | None | applied to `4*dim` FFN base |
| `norm_eps` | 1e-5 | RMSNorm eps (LayerNormContinuous uses 1e-6) |
| `axes_dim_rope` | (32,32,32) | **3-axis RoPE**, sum == dim/head |
| `axes_lens` | (300,512,512) | RoPE section max lengths |
| `text_feat_dim` | 1024 | Qwen2.5-VL hidden used as text conditioning |
| `timestep_scale` | 1.0 | |

Invariant: `hidden_size // num_attention_heads == sum(axes_dim_rope)` (dim/head = sum-of-axes).

## Module tree (with weight prefixes)

```
OmniGen2Transformer2DModel
├── x_embedder                  Linear(patch²·in_ch → hidden)         # `x_embedder.{weight,bias}`
├── ref_image_patch_embedder    Linear(patch²·in_ch → hidden)         # `ref_image_patch_embedder.{weight,bias}`
├── image_index_embedding       Parameter[5, hidden]                  # `image_index_embedding` — 5 slots for ref images
├── rope_embedder               OmniGen2RotaryPosEmbed(theta=10000)   # no learnable params (position tables)
├── time_caption_embed          Lumina2CombinedTimestepCaptionEmbedding
│   ├── time_proj               Timesteps(256, flip_sin_to_cos=T)
│   ├── timestep_embedder       (linear_1: 256→1024, linear_2: 1024→1024, SiLU between)
│   └── caption_embedder        Sequential[RMSNorm(1024), Linear(1024→hidden)]
├── noise_refiner   ModuleList(num_refiner_layers × OmniGen2TransformerBlock(modulation=True))
├── ref_image_refiner ModuleList(num_refiner_layers × OmniGen2TransformerBlock(modulation=True))
├── context_refiner ModuleList(num_refiner_layers × OmniGen2TransformerBlock(modulation=False))
├── layers          ModuleList(num_layers × OmniGen2TransformerBlock(modulation=True))
└── norm_out        LuminaLayerNormContinuous(hidden, min(hidden,1024)=1024, out_dim=patch²·out_ch)
```

## OmniGen2TransformerBlock

```
OmniGen2TransformerBlock(dim=2304, heads=24, kv_heads=8, modulation)
├── attn        Attention(Q,K,V,O; qk_norm=rms_norm; GQA 24/8; no bias)
│   # weights: attn.to_q.weight, attn.to_k.weight, attn.to_v.weight, attn.to_out.0.weight
│   # + attn.norm_q.weight, attn.norm_k.weight (RMSNorm)
├── feed_forward  LuminaFeedForward(swiglu; inner=multiple_of-rounded 4*dim*ffn_mult)
│   # linear_1 (gate), linear_2 (down), linear_3 (up)  — SwiGLU: down(swiglu(gate(x), up(x)))
├── norm1       LuminaRMSNormZero(dim,eps) if modulation else RMSNorm(dim,eps)
│   # LuminaRMSNormZero: emb→SiLU→Linear(min(dim,1024)→4·dim)→chunk 4 → (scale_msa, gate_msa, scale_mlp, gate_mlp)
│   # weights: norm1.linear.{weight,bias}, norm1.norm.weight
├── ffn_norm1   RMSNorm(dim,eps)     # norm1.norm.weight duplicated? no — this is `ffn_norm1.weight`
├── norm2       RMSNorm(dim,eps)     # `norm2.weight`
└── ffn_norm2   RMSNorm(dim,eps)     # `ffn_norm2.weight`
```

Block forward (paraphrased from Python):
1. `x_res = x`; `x, gate_msa, scale_mlp, gate_mlp = norm1(x, emb)` (or `x = norm1(x)` unmodulated)
2. `q,k,v = attn.projections(x)`; apply qk_norm; apply 3-axis RoPE to (q,k) via `rope_embedder`; GQA attention
3. `x = x_res + gate_msa * ffn_norm1(attn_out)` (modulation only)
4. `x_res = x`; `y = ffn_norm2(x) * (1 + scale_mlp)`; `y = feed_forward(y)`
5. `x = x_res + gate_mlp * norm2(y)`

## LuminaFeedForward (SwiGLU)

`h1 = linear_1(x)`, `h2 = linear_3(x)`, `out = linear_2(SiLU(h1) * h2)` — three matmuls, gated activation. Inner dim rounded up to nearest multiple of `multiple_of` (256) after applying `ffn_dim_multiplier` if given.

## 3-axis RoPE — mapping to ggml

OmniGen2 uses three RoPE sections `(t, y, x)` with dims `(32, 32, 32)` each (or `(40, 40, 40)` in some checkpoints; sum = dim/head).

ggml already has this: `ggml_rope_multi(...)` with `mode = GGML_ROPE_TYPE_MROPE` and `sections = {32, 32, 32, 0}` (last section ignored). See `2-contract/ggml/include/ggml.h:1857-1883`.

**No new ggml op needed.** The section-array shape maps 1-1 to OmniGen2's `axes_dim_rope`. `axes_lens` is the position-table max length per section, materialized outside the op by whoever generates the position tensor `b` fed into `ggml_rope_multi`.

## Conditioning input (from caller)

The Python model consumes text conditioning as `text_hidden_states: [B, L_text, text_feat_dim=1024]` (from Qwen2.5-VL). sd.cpp already has a Qwen2.5-VL text encoder (used for `qwen_image`) — we can reuse it for `omnigen2`. `hidden_size=2304` and `text_feat_dim=1024` — the `caption_embedder` linear projects Qwen text hidden into transformer width.

## Sampler + VAE

- Scheduler: `FlowMatchEulerDiscreteScheduler` — sd.cpp already supports (used by FLUX / Qwen-Image).
- VAE: `AutoencoderKL` — the same one Qwen-Image / other DiTs use. Reuse existing sd.cpp VAE.

## OmniGen2AttnProcessor

Straightforward multi-head attention with:
- RMS `qk_norm` before attention
- 3-axis RoPE on q and k (before attention)
- GQA repeat-KV to match Q head count
- Standard scaled-dot-product-attention

Nothing exotic beyond the RoPE section shape.

## Milestone map

- **M1 (this PR):** arch notes, C++ skeleton compiles, `VERSION_OMNIGEN2` enum, converter Python skeleton reads tensor names.
- M2: implement Block forward (attention + RoPE + FFN); norm_out + patch unembed.
- M3: end-to-end forward returning a same-shape latent; Metal smoke against random noise.
- M4: converter emits a valid GGUF sd.cpp `model_load` can accept; VAE + text_encoder wiring.
- M5: full sampler loop, one image out, MaskScore MC on Metal (roundtrip vs Python reference).

## Missing pieces (still) — none blocking

- `image_index_embedding` is a raw `Parameter[5, hidden]` — sd.cpp doesn't have a "raw parameter" primitive that isn't a weight; we'll wrap it as a `Linear`-like block or store it directly via `params_ctx`. Trivial.
- `Timesteps` sinusoidal embedding — sd.cpp already has this pattern in `qwen_image.hpp`. Reuse.
- Qwen2.5-VL text encoder shared with `qwen_image` — dispatch existing conditioner.

No new ggml ops required. No arch feature genuinely absent from sd.cpp's building blocks.
