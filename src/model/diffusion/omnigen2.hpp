#ifndef __SD_MODEL_DIFFUSION_OMNIGEN2_HPP__
#define __SD_MODEL_DIFFUSION_OMNIGEN2_HPP__

// OmniGen2 (Lumina2-derived DiT + dual-refiner + 3-axis RoPE) skeleton.
// Milestone 1: declarations compile against sd.cpp headers; forward() is a
// well-typed stub. Not runnable yet. See docs/omnigen2-arch-notes.md.
//
// Config defaults (from VectorSpaceLab/OmniGen2 transformer_omnigen2.py):
//   patch=2, in_ch=16, hidden=2304, layers=26, refiner_layers=2,
//   heads=24, kv_heads=8, multiple_of=256, axes_dim_rope=(32,32,32),
//   axes_lens=(300,512,512), text_feat_dim=1024.
//
// 3-axis RoPE uses ggml_rope_multi with GGML_ROPE_TYPE_MROPE and
// sections={axes_dim_rope[0], axes_dim_rope[1], axes_dim_rope[2], 0}.

#include <memory>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/common/block.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace OmniGen2 {

    // Container for named sub-blocks — exposes GGMLBlock::blocks publicly for
    // composite groupings that don't warrant their own named class
    // (nn.ModuleList / nn.Sequential analogs).
    struct BlockGroup : public GGMLBlock {
    public:
        using GGMLBlock::blocks;
    };

    constexpr int OMNIGEN2_GRAPH_SIZE = 20480;

    struct OmniGen2Config {
        int patch_size              = 2;
        int64_t in_channels         = 16;
        int64_t out_channels        = 16;
        int64_t hidden_size         = 2304;
        int num_layers              = 26;
        int num_refiner_layers      = 2;
        int64_t num_attention_heads = 24;
        int64_t num_kv_heads        = 8;
        int64_t multiple_of         = 256;
        float ffn_dim_multiplier    = 0.0f;   // 0 => unset (use 4*hidden as inner base)
        float norm_eps              = 1e-5f;
        std::vector<int> axes_dim   = {32, 32, 32};
        std::vector<int> axes_lens  = {300, 512, 512};
        int64_t text_feat_dim       = 1024;
        float timestep_scale        = 1.0f;
        int theta                   = 10000;
        int max_ref_images          = 5;

        static OmniGen2Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                   const std::string& prefix) {
            OmniGen2Config config;
            config.num_layers          = 0;
            config.num_refiner_layers  = 0;
            int noise_refiner_count    = 0;
            int ref_image_refiner_count = 0;
            int context_refiner_count  = 0;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, prefix)) continue;
                size_t p = name.find("layers.");
                if (p != std::string::npos) {
                    auto items = split_string(name.substr(p), '.');
                    if (items.size() > 1) {
                        int idx = atoi(items[1].c_str());
                        if (idx + 1 > config.num_layers) config.num_layers = idx + 1;
                    }
                }
                if (name.find("noise_refiner.") != std::string::npos) {
                    auto q = name.find("noise_refiner.");
                    auto items = split_string(name.substr(q), '.');
                    if (items.size() > 1) {
                        int idx = atoi(items[1].c_str());
                        if (idx + 1 > noise_refiner_count) noise_refiner_count = idx + 1;
                    }
                }
                if (name.find("ref_image_refiner.") != std::string::npos) {
                    auto q = name.find("ref_image_refiner.");
                    auto items = split_string(name.substr(q), '.');
                    if (items.size() > 1) {
                        int idx = atoi(items[1].c_str());
                        if (idx + 1 > ref_image_refiner_count) ref_image_refiner_count = idx + 1;
                    }
                }
                if (name.find("context_refiner.") != std::string::npos) {
                    auto q = name.find("context_refiner.");
                    auto items = split_string(name.substr(q), '.');
                    if (items.size() > 1) {
                        int idx = atoi(items[1].c_str());
                        if (idx + 1 > context_refiner_count) context_refiner_count = idx + 1;
                    }
                }
            }
            // All three refiner stacks should share num_refiner_layers.
            config.num_refiner_layers = std::max({noise_refiner_count,
                                                   ref_image_refiner_count,
                                                   context_refiner_count});
            LOG_DEBUG("omnigen2: num_layers=%d, num_refiner_layers=%d",
                      config.num_layers, config.num_refiner_layers);
            return config;
        }
    };

    // -- Lumina2 building blocks ---------------------------------------------

    // LuminaRMSNormZero: adaptive RMSNorm producing (x_normed, gate_msa, scale_mlp, gate_mlp).
    // linear projects a conditioning embedding into 4*dim, chunked into the four modulation params.
    struct LuminaRMSNormZero : public GGMLBlock {
    public:
        LuminaRMSNormZero(int64_t embedding_dim, float norm_eps) {
            int64_t cond_dim = std::min<int64_t>(embedding_dim, 1024);
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(cond_dim, 4 * embedding_dim, /*bias*/ true));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(embedding_dim, norm_eps));
        }
        // Forward emitted in M2.
    };

    // LuminaLayerNormContinuous: adaptive layer-norm with optional out projection.
    struct LuminaLayerNormContinuous : public GGMLBlock {
    public:
        LuminaLayerNormContinuous(int64_t embedding_dim,
                                   int64_t conditioning_embedding_dim,
                                   int64_t out_dim = 0,
                                   float eps       = 1e-6f) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(conditioning_embedding_dim, embedding_dim, /*bias*/ true));
            blocks["norm"]     = std::shared_ptr<GGMLBlock>(new LayerNorm(embedding_dim, eps, /*elementwise_affine*/ false));
            if (out_dim > 0) {
                blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(embedding_dim, out_dim, /*bias*/ true));
            }
        }
    };

    // LuminaFeedForward: SwiGLU (linear_1 + linear_3 gate → SiLU → * → linear_2 down).
    // inner_dim rounded up to nearest multiple_of after applying ffn_dim_multiplier.
    struct LuminaFeedForward : public GGMLBlock {
    public:
        LuminaFeedForward(int64_t dim,
                          int64_t inner_dim_base,
                          int64_t multiple_of        = 256,
                          float ffn_dim_multiplier   = 0.0f) {
            int64_t inner_dim = inner_dim_base;
            if (ffn_dim_multiplier > 0.0f) {
                inner_dim = static_cast<int64_t>(ffn_dim_multiplier * inner_dim);
            }
            inner_dim = multiple_of * ((inner_dim + multiple_of - 1) / multiple_of);
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim, /*bias*/ false));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim, /*bias*/ false));
            blocks["linear_3"] = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim, /*bias*/ false));
        }
    };

    // Lumina2CombinedTimestepCaptionEmbedding: sinusoidal timestep + text caption embed.
    // Weight paths: time_proj (no params), timestep_embedder.{linear_1,linear_2},
    //               caption_embedder.0 (RMSNorm), caption_embedder.1 (Linear).
    struct Lumina2CombinedTimestepCaptionEmbedding : public GGMLBlock {
    public:
        Lumina2CombinedTimestepCaptionEmbedding(int64_t hidden_size,
                                                 int64_t text_feat_dim,
                                                 float norm_eps = 1e-5f) {
            int64_t time_embed_out = std::min<int64_t>(hidden_size, 1024);
            // timestep_embedder: 256 -> time_embed_out (with SiLU in between the two Linears).
            {
                auto emb_block          = std::make_shared<BlockGroup>();
                emb_block->blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(256, time_embed_out, /*bias*/ true));
                emb_block->blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_out, time_embed_out, /*bias*/ true));
                blocks["timestep_embedder"] = emb_block;
            }
            // caption_embedder is a Sequential in Python; we split as .0 (RMSNorm) and .1 (Linear).
            {
                auto cap_block                = std::make_shared<BlockGroup>();
                cap_block->blocks["0"]        = std::shared_ptr<GGMLBlock>(new RMSNorm(text_feat_dim, norm_eps));
                cap_block->blocks["1"]        = std::shared_ptr<GGMLBlock>(new Linear(text_feat_dim, hidden_size, /*bias*/ true));
                blocks["caption_embedder"]    = cap_block;
            }
        }
    };

    // -- OmniGen2 transformer block ------------------------------------------

    // OmniGen2TransformerBlock: RMSNormZero (or plain RMSNorm) → attn → FFN.
    // attn is GQA (num_kv_heads < num_attention_heads); qk_norm is RMSNorm on q and k.
    // Uses 3-axis RoPE (ggml_rope_multi with MROPE mode).
    struct OmniGen2TransformerBlock : public GGMLBlock {
    public:
        bool modulation;

        OmniGen2TransformerBlock(int64_t dim,
                                  int64_t num_attention_heads,
                                  int64_t num_kv_heads,
                                  int64_t multiple_of,
                                  float ffn_dim_multiplier,
                                  float norm_eps,
                                  bool modulation)
            : modulation(modulation) {
            int64_t head_dim = dim / num_attention_heads;

            // Attention (GQA). to_q width = heads * head_dim; to_k/to_v width = kv_heads * head_dim.
            {
                auto attn_block = std::make_shared<BlockGroup>();
                attn_block->blocks["to_q"]      = std::shared_ptr<GGMLBlock>(new Linear(dim, num_attention_heads * head_dim, /*bias*/ false));
                attn_block->blocks["to_k"]      = std::shared_ptr<GGMLBlock>(new Linear(dim, num_kv_heads * head_dim, /*bias*/ false));
                attn_block->blocks["to_v"]      = std::shared_ptr<GGMLBlock>(new Linear(dim, num_kv_heads * head_dim, /*bias*/ false));
                attn_block->blocks["norm_q"]    = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, norm_eps));
                attn_block->blocks["norm_k"]    = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, norm_eps));
                // to_out is a Sequential[Linear, Dropout]; only Linear at .0 has weights.
                auto out_block                  = std::make_shared<BlockGroup>();
                out_block->blocks["0"]          = std::shared_ptr<GGMLBlock>(new Linear(num_attention_heads * head_dim, dim, /*bias*/ false));
                attn_block->blocks["to_out"]    = out_block;
                blocks["attn"]                  = attn_block;
            }

            // Feed-forward (SwiGLU).
            blocks["feed_forward"] = std::shared_ptr<GGMLBlock>(new LuminaFeedForward(dim, 4 * dim, multiple_of, ffn_dim_multiplier));

            // Norms — modulation branch swaps the first norm.
            if (modulation) {
                blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LuminaRMSNormZero(dim, norm_eps));
            } else {
                blocks["norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
            }
            blocks["ffn_norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
            blocks["norm2"]     = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
            blocks["ffn_norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
        }
        // Forward emitted in M2.
    };

    // -- Top-level model -----------------------------------------------------

    struct OmniGen2Model : public GGMLBlock {
    public:
        OmniGen2Config config;

        OmniGen2Model() = default;
        explicit OmniGen2Model(const OmniGen2Config& cfg)
            : config(cfg) {
            int64_t patch_in = static_cast<int64_t>(cfg.patch_size * cfg.patch_size) * cfg.in_channels;

            blocks["x_embedder"]               = std::shared_ptr<GGMLBlock>(new Linear(patch_in, cfg.hidden_size, /*bias*/ true));
            blocks["ref_image_patch_embedder"] = std::shared_ptr<GGMLBlock>(new Linear(patch_in, cfg.hidden_size, /*bias*/ true));

            blocks["time_caption_embed"] = std::shared_ptr<GGMLBlock>(
                new Lumina2CombinedTimestepCaptionEmbedding(cfg.hidden_size, cfg.text_feat_dim, cfg.norm_eps));

            // Refiner stacks — three groups (noise, ref_image, context).
            {
                auto noise = std::make_shared<BlockGroup>();
                for (int i = 0; i < cfg.num_refiner_layers; ++i) {
                    noise->blocks[std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                        new OmniGen2TransformerBlock(cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads,
                                                      cfg.multiple_of, cfg.ffn_dim_multiplier, cfg.norm_eps,
                                                      /*modulation*/ true));
                }
                blocks["noise_refiner"] = noise;
            }
            {
                auto ref_img = std::make_shared<BlockGroup>();
                for (int i = 0; i < cfg.num_refiner_layers; ++i) {
                    ref_img->blocks[std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                        new OmniGen2TransformerBlock(cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads,
                                                      cfg.multiple_of, cfg.ffn_dim_multiplier, cfg.norm_eps,
                                                      /*modulation*/ true));
                }
                blocks["ref_image_refiner"] = ref_img;
            }
            {
                auto ctx = std::make_shared<BlockGroup>();
                for (int i = 0; i < cfg.num_refiner_layers; ++i) {
                    ctx->blocks[std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                        new OmniGen2TransformerBlock(cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads,
                                                      cfg.multiple_of, cfg.ffn_dim_multiplier, cfg.norm_eps,
                                                      /*modulation*/ false));
                }
                blocks["context_refiner"] = ctx;
            }

            // Main transformer stack.
            {
                auto layers = std::make_shared<BlockGroup>();
                for (int i = 0; i < cfg.num_layers; ++i) {
                    layers->blocks[std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                        new OmniGen2TransformerBlock(cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads,
                                                      cfg.multiple_of, cfg.ffn_dim_multiplier, cfg.norm_eps,
                                                      /*modulation*/ true));
                }
                blocks["layers"] = layers;
            }

            // Output norm + patch unembed.
            int64_t out_dim = static_cast<int64_t>(cfg.patch_size * cfg.patch_size) * cfg.out_channels;
            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(
                new LuminaLayerNormContinuous(cfg.hidden_size, std::min<int64_t>(cfg.hidden_size, 1024), out_dim));

            // image_index_embedding: raw parameter[max_ref_images, hidden_size]. Registered via params_ctx
            // during init in M2 (needs a raw-parameter primitive; not yet in the block system).
        }
    };

    // Runner — matches the DiffusionModelRunner interface. compute() is a stub for M1.
    struct OmniGen2Runner : public DiffusionModelRunner {
    public:
        OmniGen2Config config;
        OmniGen2Model omnigen2;
        SDVersion version;

        OmniGen2Runner(ggml_backend_t backend,
                        const String2TensorStorage& tensor_storage_map      = {},
                        const std::string prefix                            = "",
                        SDVersion version                                   = SDVersion::VERSION_OMNIGEN2,
                        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr,
                        const char* /*model_args*/                          = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(OmniGen2Config::detect_from_weights(tensor_storage_map, prefix)),
              version(version) {
            omnigen2 = OmniGen2Model(config);
            omnigen2.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "omnigen2";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                const std::string& prefix) override {
            omnigen2.get_param_tensors(tensors, prefix);
        }

        // Milestone 1 stub: signals that the runner exists but forward isn't wired.
        // Milestone 2 will implement the block forward + attention + RoPE + FFN.
        // Milestone 3 will wire the full graph and produce a same-shape latent.
        sd::Tensor<float> compute(int /*n_threads*/,
                                   const DiffusionParams& diffusion_params) override {
            LOG_WARN("omnigen2: forward() not implemented (Milestone 1 skeleton); returning input as passthrough");
            if (diffusion_params.x != nullptr) {
                return *diffusion_params.x;
            }
            return sd::Tensor<float>();
        }
    };

}  // namespace OmniGen2

#endif  // __SD_MODEL_DIFFUSION_OMNIGEN2_HPP__
