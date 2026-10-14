#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <chrono>
#include <regex>
#include <tuple>
#include <cmath>
#include <limits>
#include "std_compat/memory.h"
#include "std_compat/utility.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

// Stage handlers — adding a new stage: create stage_<name>.h, include it here,
// push a make_unique<XStageKind>() in make_stage_kinds() below.
#include "stage_kind.h"
#include "archive.h"
#include "stage_lorenzo.h"
#include "stage_quantizer.h"
#include "stage_diff.h"
#include "stage_rle.h"
#include "stage_zigzag.h"
#include "stage_negabinary.h"
#include "stage_bitshuffle.h"
#include "stage_bitpack.h"
#include "stage_rze.h"
#include "stage_tiled_lorenzo.h"
#include "stage_adaptive_bitpack.h"
#include "stage_rre.h"
#include "stage_rare.h"
#include "stage_raze.h"
#include "stage_clog.h"
#include "stage_hclog.h"
#include "stage_gpulz.h"
#include "stage_tupl.h"
#include "stage_ans.h"
#if LIBPRESSIO_FZGM_HAS_ADM
#include "stage_adm.h"
#endif
#include "stage_bitplane_rze.h"
#if LIBPRESSIO_FZGM_HAS_SZX
#include "stage_szx.h"
#endif
#if LIBPRESSIO_FZGM_HAS_SZP
#include "stage_szp.h"
#endif
#include "stage_huffman.h"
#include "stage_adaptive_lorenzo.h"
#include "stage_log_transform.h"
#include "stage_ginterp.h"
#include "stage_roibin_split.h"
#if LIBPRESSIO_FZGM_HAS_CDF97
#include "stage_cdf97.h"
#endif
#if LIBPRESSIO_FZGM_HAS_SPECK2D
#include "stage_speck2d.h"
#endif
#if LIBPRESSIO_FZGM_HAS_CDF97_OUTLIER_CORRECT
#include "stage_outlier_correct.h"
#endif

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

using namespace libpressio::compressors;  // brings libpressio_compressor_plugin, compressor_plugins, etc.

// =============================================================================
// Pipeline-level helpers
// =============================================================================

static const std::map<std::string, fz::MemoryStrategy> MEMORY_STRATEGIES {
    {"preallocate", fz::MemoryStrategy::PREALLOCATE},
    {"minimal",     fz::MemoryStrategy::MINIMAL},
};

static const std::map<std::string, fz::ErrorBoundMode> ERROR_BOUND_MODES {
    {"abs",  fz::ErrorBoundMode::ABS},
    {"rel",  fz::ErrorBoundMode::REL},
    {"noa",  fz::ErrorBoundMode::NOA},
    {"prel", fz::ErrorBoundMode::PREL},
};

static const std::map<std::string, fz::SpecializationPolicy> SPECIALIZATION_POLICIES {
    {"off",  fz::SpecializationPolicy::Off},
    {"auto", fz::SpecializationPolicy::Auto},
};

template <class T>
static std::vector<std::string> map_keys(std::map<std::string, T> const& m) {
    std::vector<std::string> keys;
    keys.reserve(m.size());
    for (auto const& kv : m) keys.push_back(kv.first);
    return keys;
}

struct ParsedConnection { std::string to, from, port; };

static ParsedConnection parse_connection(const std::string& s) {
    static const std::regex re(R"(^\s*(\S+)\s*<-\s*(\S+?)(?::(\S+))?\s*$)");
    std::smatch m;
    if(!std::regex_match(s, m, re))
        throw std::invalid_argument("Invalid connection (expected 'sN <- sM[:port]'): " + s);
    return {m[1].str(), m[2].str(), m[3].str()};
}

// =============================================================================
// RAII helpers
// =============================================================================

struct CudaStream {
    cudaStream_t s = nullptr;
    CudaStream()  { cudaStreamCreate(&s); }
    ~CudaStream() { if(s) cudaStreamDestroy(s); }
    CudaStream(CudaStream&& o) noexcept : s(std::exchange(o.s, nullptr)) {}
    CudaStream& operator=(CudaStream&& o) noexcept {
        if(this != &o) { if(s) cudaStreamDestroy(s); s = std::exchange(o.s, nullptr); }
        return *this;
    }
    CudaStream(CudaStream const&)            { cudaStreamCreate(&s); }
    CudaStream& operator=(CudaStream const&) { if(!s) cudaStreamCreate(&s); return *this; }
    operator cudaStream_t() const { return s; }
};

struct OutlierStageRef {
    fz::Stage*  ptr;
    std::string indices_key;  // output port whose size / 4 == outlier count
};

struct TerminalOutput {
    std::string   metric_key;  // "fzgpumodules:sN:output:<port>"
    pressio_dtype dtype;
};

struct PipelineState {
    std::unique_ptr<fz::Pipeline>              ptr;
    bool   dirty     = true;
    size_t last_size = 0;
    std::vector<std::pair<std::string, OutlierStageRef>> outlier_stages;   // sid → stage ref
    std::vector<TerminalOutput>                          terminal_outputs;  // ordered terminal ports

    PipelineState() = default;
    PipelineState(PipelineState&&) = default;
    PipelineState& operator=(PipelineState&&) = default;
    PipelineState(PipelineState const&)            : ptr(nullptr), dirty(true), last_size(0) {}
    PipelineState& operator=(PipelineState const&) {
        ptr.reset(); dirty = true; last_size = 0;
        outlier_stages.clear(); terminal_outputs.clear();
        return *this;
    }
};

// =============================================================================
// Pipeline construction helpers (shared by both plugin classes)
// =============================================================================

// Apply common per-pipeline settings after construction.
// Pass dims=nullptr in config-file mode (the TOML sets dims internally).
static void configure_pipeline_base(fz::Pipeline& p,
                                     const std::array<size_t,3>* dims,
                                     int num_streams,
                                     fz::SpecializationPolicy specialization_policy) {
    // Decompress ownership is chosen at the call site via decompressOwned(); no flag needed.
    if (dims) p.setDims(*dims);
    if (num_streams > 1) p.setNumStreams(num_streams);
    p.setSpecializationPolicy(specialization_policy);
}

// Run warmup + graph capture after finalize(), if graph_mode is set.
static void maybe_capture_graph(fz::Pipeline& p, cudaStream_t stream, bool graph_mode) {
    if (graph_mode) {
        p.warmup(stream);
        p.captureGraph(stream);
    }
}

static pressio_dtype fz_to_pressio_dtype(fz::DataType dt) {
    switch (dt) {
        case fz::DataType::UINT8:   return pressio_uint8_dtype;
        case fz::DataType::UINT16:  return pressio_uint16_dtype;
        case fz::DataType::UINT32:  return pressio_uint32_dtype;
        case fz::DataType::UINT64:  return pressio_uint64_dtype;
        case fz::DataType::INT8:    return pressio_int8_dtype;
        case fz::DataType::INT16:   return pressio_int16_dtype;
        case fz::DataType::INT32:   return pressio_int32_dtype;
        case fz::DataType::INT64:   return pressio_int64_dtype;
        case fz::DataType::FLOAT32: return pressio_float_dtype;
        case fz::DataType::FLOAT64: return pressio_double_dtype;
        default:                    return pressio_byte_dtype;
    }
}

// =============================================================================
// Main plugin class
// =============================================================================

class fzgpumodules_plugin : public libpressio_compressor_plugin {
public:

    fzgpumodules_plugin() : stage_kinds_(make_stage_kinds()) {}

    // =========================================================================
    // Stage registry factory
    // =========================================================================

    static std::vector<std::unique_ptr<StageKind>> make_stage_kinds() {
        std::vector<std::unique_ptr<StageKind>> v;
        v.push_back(std::make_unique<LorenzoStageKind>());
        v.push_back(std::make_unique<QuantizerStageKind>());
        v.push_back(std::make_unique<DiffStageKind>());
        v.push_back(std::make_unique<RLEStageKind>());
        v.push_back(std::make_unique<ZigzagStageKind>());
        v.push_back(std::make_unique<NegabinaryStageKind>());
        v.push_back(std::make_unique<BitshuffleStageKind>());
        v.push_back(std::make_unique<BitpackStageKind>());
        v.push_back(std::make_unique<RZEStageKind>());
        v.push_back(std::make_unique<TiledLorenzoStageKind>());
        v.push_back(std::make_unique<AdaptiveBitpackStageKind>());
        v.push_back(std::make_unique<RREStageKind>());
        v.push_back(std::make_unique<RAREStageKind>());
        v.push_back(std::make_unique<RAZEStageKind>());
        v.push_back(std::make_unique<CLOGStageKind>());
        v.push_back(std::make_unique<HCLOGStageKind>());
        v.push_back(std::make_unique<GPULZStageKind>());
        v.push_back(std::make_unique<TUPLStageKind>());
        v.push_back(std::make_unique<ANSStageKind>());
#if LIBPRESSIO_FZGM_HAS_ADM
        v.push_back(std::make_unique<ADMStageKind>());
#endif
        v.push_back(std::make_unique<BitplaneRZEStageKind>());
#if LIBPRESSIO_FZGM_HAS_SZX
        v.push_back(std::make_unique<SZxStageKind>());
#endif
#if LIBPRESSIO_FZGM_HAS_SZP
        v.push_back(std::make_unique<SZpStageKind>());
#endif
        v.push_back(std::make_unique<HuffmanStageKind>());
        v.push_back(std::make_unique<AdaptiveLorenzoStageKind>());
        v.push_back(std::make_unique<LogTransformStageKind>());
        v.push_back(std::make_unique<GInterpStageKind>());
        v.push_back(std::make_unique<ROIBinSplitStageKind>());
#if LIBPRESSIO_FZGM_HAS_CDF97
        v.push_back(std::make_unique<Cdf97StageKind>());
#endif
#if LIBPRESSIO_FZGM_HAS_SPECK2D
        v.push_back(std::make_unique<Speck2DStageKind>());
#endif
#if LIBPRESSIO_FZGM_HAS_CDF97_OUTLIER_CORRECT
        v.push_back(std::make_unique<Cdf97OutlierCorrectStageKind>());
#endif
        return v;
    }

    // =========================================================================
    // Options interface
    // =========================================================================

    struct pressio_options get_options_impl() const override {
        struct pressio_options options;

        set(options, "pressio:abs", error_bound_abs_);
        set(options, "pressio:rel", error_bound_rel_);
        set(options, "fzgpumodules:memory_strategy",   memory_strategy_);
        set(options, "fzgpumodules:memory_multiplier", memory_multiplier_);
        set(options, "fzgpumodules:error_bound_mode",  error_bound_mode_);
        set(options, "fzgpumodules:dims", pressio_data(dims_.begin(), dims_.end()));
        set(options, "fzgpumodules:num_streams", num_streams_);
        set(options, "fzgpumodules:graph_mode",  graph_mode_);
        set(options, "fzgpumodules:specialization", specialization_);
        set(options, "fzgpumodules:stages",               stages_);
        set(options, "fzgpumodules:connections",          connections_);
        set(options, "fzgpumodules:primary_source",       primary_source_);
        set(options, "fzgpumodules:config_file",          config_file_);
        set(options, "fzgpumodules:expose_stage_outputs", expose_stage_outputs_);

        if (config_file_.empty()) {
            for_each_stage([&](StageKind& sk, const std::string& sid, const std::string& tok) {
                sk.populate_options(options, sid, tok);
            });
        }

        return options;
    }

    struct pressio_options get_configuration_impl() const override {
        struct pressio_options options;
        set(options, "pressio:thread_safe",  pressio_thread_safety_serialized);
        set(options, "pressio:stability",    "experimental");
        set(options, "fzgpumodules:memory_strategy",  map_keys(MEMORY_STRATEGIES));
        set(options, "fzgpumodules:error_bound_mode", map_keys(ERROR_BOUND_MODES));
        set(options, "fzgpumodules:specialization", map_keys(SPECIALIZATION_POLICIES));
        std::vector<std::string> optional_stages;
#if LIBPRESSIO_FZGM_HAS_SZX
        optional_stages.emplace_back("szx");
#endif
#if LIBPRESSIO_FZGM_HAS_CDF97
        optional_stages.emplace_back("cdf97");
#endif
#if LIBPRESSIO_FZGM_HAS_SPECK2D
        optional_stages.emplace_back("speck2d");
#endif
#if LIBPRESSIO_FZGM_HAS_CDF97_OUTLIER_CORRECT
        optional_stages.emplace_back("cdf97_outlier_correct");
#endif
        set(options, "fzgpumodules:optional_stages", optional_stages);
        set(options, "predictors:error_dependent",
            std::vector<std::string>{"fzgpumodules:stages", "fzgpumodules:connections"});
        set(options, "predictors:error_agnostic",
            std::vector<std::string>{"fzgpumodules:stages", "fzgpumodules:connections"});
        set(options, "predictors:runtime",
            std::vector<std::string>{"fzgpumodules:memory_strategy", "fzgpumodules:num_streams",
                                      "fzgpumodules:specialization"});
        return options;
    }

    struct pressio_options get_documentation_impl() const override {
        struct pressio_options options;

        set(options, "pressio:description",
            "FZGPUModules GPU compressor with user-composable pipeline stages.\n\n"
            "Configure the pipeline by setting `fzgpumodules:stages` (ordered list of stage tokens) "
            "and `fzgpumodules:connections` (list of wiring strings). "
            "Set `fzgpumodules:dims` for multi-dimensional data (default: infer 1-D from input size).");
        set(options, "pressio:abs",
            "Absolute error bound. Also holds the bound fraction for fzgpumodules:error_bound_mode "
            "'noa' (value-range fraction) and 'prel' (fraction of max|data|).");
        set(options, "pressio:rel",
            "Point-wise relative error bound (used when fzgpumodules:error_bound_mode is 'rel')");
        set(options, "fzgpumodules:error_bound_mode",
            "Error bound mode: abs (default), rel, noa, or prel.\n"
            "  abs  — absolute error bound (pressio:abs).\n"
            "  rel  — point-wise relative bound (pressio:rel). Exact for QuantizerStage-backed "
            "stages. Stages that quantize a fused prediction residual against one global "
            "tolerance (e.g. the lorenzo:float/double tokens, which build LorenzoQuantStage) "
            "cannot honor an exact point-wise bound: they treat 'rel' as a deprecated alias "
            "for 'prel' and log a warning. Use 'prel' explicitly on those stages to silence it.\n"
            "  noa  — value-range relative bound (pressio:abs as a fraction of the data's value range).\n"
            "  prel — pseudo-relative: abs_eb = pressio:abs * max(|data|), applied as a single "
            "absolute bound. This is the honest name for what 'rel' silently did on fused "
            "prediction stages before the REL/PREL split.");
        set(options, "fzgpumodules:memory_strategy",
            "GPU memory allocation strategy: preallocate or minimal (default)");
        set(options, "fzgpumodules:memory_multiplier",
            "Memory allocation multiplier (e.g. 3.0 = 3x input size)");
        set(options, "fzgpumodules:dims",
            "Spatial dimensions [x, y, z] (x = fast). Default {0,1,1} infers 1-D from input size.");
        set(options, "fzgpumodules:num_streams",
            "Number of parallel CUDA streams (default: 1)");
        set(options, "fzgpumodules:graph_mode",
            "Enable CUDA Graph capture for repeated compression (default: false). "
            "Eliminates per-call CPU dispatch overhead after a one-time warmup+capture on "
            "the first compress() call. Requires memory_strategy=preallocate. "
            "Incompatible with rze stage. "
            "In graph mode, compress() may be called any number of times without decompressing "
            "first — each call replays the captured graph and overwrites the internal DAG "
            "buffers. decompress() reads from those live GPU buffers (the most recent "
            "compress() result); the compressed_input bytes are not parsed. "
            "Cross-machine or cross-process decompression is not supported in graph mode.");
        set(options, "fzgpumodules:specialization",
            "Pipeline Specialization policy: off (default) or auto. Auto installs admitted "
            "implementations for supported stage combinations and otherwise uses staged execution. "
            "It is not a per-input profitability predictor and is mutually exclusive with CUDA Graph mode. "
            "FZ_SPECIALIZE overrides the policy at pipeline-build time. "
            "Inspect specialization_* metrics after encode/decode to confirm installation. "
            "Use fzgpumodules:specialization in Python and C++ configurations.");
        set(options, "fzgpumodules:stages",
            "Ordered list of stage tokens.\n"
            "Quantizing:        lorenzo:float:uint8/uint16, lorenzo:double:uint16/uint32,\n"
            "                   quantizer:float:uint16/uint32, quantizer:double:uint16/uint32.\n"
            "Integer Lorenzo:   lorenzo:int8, lorenzo:int16, lorenzo:int32, lorenzo:int64.\n"
            "Tiled predictor:   tiled_lorenzo:int16, tiled_lorenzo:int32 (dimension-aware,\n"
            "                   separable Lorenzo; tile-major output for AdaptiveBitpack).\n"
#if LIBPRESSIO_FZGM_HAS_SZX
            "Optional lossy:    szx:float, szx:double (abs/noa error only).\n"
#endif
#if LIBPRESSIO_FZGM_HAS_SZP
            "Legacy lossy:      szp:float, szp:double (abs/noa error only).\n"
#endif
            "Adaptive Lorenzo:  adaptive_lorenzo:int16/32 (lossless, 1 in/3 out fwd).\n"
            "Interpolation:     ginterp (cuSZ-Hi interpolation, 2-D/3-D only, 1 in/4 out fwd).\n"
            "Transforms:        zigzag:int8/16/32/64, negabinary:int8/16/32/64"
#if LIBPRESSIO_FZGM_HAS_ADM
            ", adm:uint16/uint32"
#endif
            ",\n"
            "                   log_transform (float, exact point-wise rel bound, 1 in/4 out fwd).\n"
            "Difference:        diff:float, diff:double, diff:uint8/16/32, diff:int32/64,\n"
            "                   diff:int8:uint8, diff:int16:uint16, diff:int32:uint32, diff:int64:uint64.\n"
            "Entropy coders:    rle:uint8/16/32, bitpack:uint8/16/32, adaptive_bitpack:int16/32,\n"
            "                   huffman:uint8/16/32, ans.\n"
            "LC byte reducers:  rze, rre, rare, raze, clog, hclog (all take chunk_size/word_size).\n"
            "Byte-level:        bitshuffle, gpulz, tupl, bitplane_rze (fixed uint16 input).\n"
            "Structural:        roibin_split:float/double (needs fzgpumodules:sN:peaks_file,\n"
            "                   1 in/3 out fwd).\n"
#if LIBPRESSIO_FZGM_HAS_CDF97
            "Optional transform: cdf97:float/double (CDF 9/7 DWT; coefficient-domain quantization).\n"
#endif
#if LIBPRESSIO_FZGM_HAS_SPECK2D
            "Optional coder:    speck2d (2-D bit-plane coder).\n"
#endif
#if LIBPRESSIO_FZGM_HAS_CDF97_OUTLIER_CORRECT
            "Optional correction: cdf97_outlier_correct (sparse correction against raw input;\n"
            "                   requires an external input binding and matching bound).\n"
#endif
            "The stages marked 'N in/M out fwd' above have a port count that swaps with\n"
            "direction (M in/N out on decompress) — Pipeline::buildInverseDAG() reconciles\n"
            "this transparently, nothing special to configure.\n"
            "See each stage's fzgpumodules:sN:<param> documentation (get_documentation) for "
            "per-stage tuning knobs, or the TOML config_file path for stages not listed here "
            "(Merge, GPULZ split_mode — these need the stage's forward port count to *grow* "
            "after construction based on a config value, which addRawStage() doesn't support "
            "since it captures port counts once, right after addStage() returns).");
        set(options, "fzgpumodules:connections",
            "Wiring strings of the form 'sN <- sM' or 'sN <- sM:port'. The reserved source name "
            "'__external__' binds the pipeline's raw input directly to sN's next input port "
            "(Pipeline::bindExternalInput()) instead of another stage's output — e.g. "
            "'s2 <- __external__' — for a stage that needs the untouched original input on one "
            "port while another port connects normally. See "
            "fzgpumodules:primary_source when this creates more than one source stage.");
        set(options, "fzgpumodules:primary_source",
            "Stage id (e.g. 's2') whose inverse output decompress() returns, for pipelines with "
            "more than one source stage (see the '__external__' connection above). Empty string "
            "(default) uses the sole/first-discovered source, which is almost never what you want "
            "once a second source exists — set this explicitly whenever fzgpumodules:connections "
            "contains a '__external__' entry alongside a stage that also has a real connection.");
        set(options, "fzgpumodules:config_file",
            "Path to a TOML pipeline config file. When non-empty, the file is the source of "
            "truth for stages, connections, dims, and per-stage error bounds. Options "
            "fzgpumodules:stages, fzgpumodules:connections, fzgpumodules:dims, and pressio:abs "
            "are ignored. fzgpumodules:graph_mode, fzgpumodules:num_streams, and "
            "fzgpumodules:memory_multiplier may still be set via libpressio options. "
            "Per-stage outlier_count metrics are not available in config-file mode. "
            "Set to empty string to revert to the stages/connections API. "
            "See docs/FZGPUModulesUsage.md for the TOML format.");
        set(options, "fzgpumodules:expose_stage_outputs",
            "When true, each unconnected stage output port is exposed as a pressio_data "
            "metric under 'fzgpumodules:sN:output:<port>' after compress(). "
            "For a single-stage pipeline this lets you inspect the raw data on both "
            "sides of the transformation. Not available in config_file mode.");
        set(options, "fzgpumodules:peak_memory",
            "[Output] Peak GPU memory usage from last compression (bytes)");
        set(options, "fzgpumodules:execution_time_us",
            "[Output] Execution time from last compression (microseconds). Stream-synchronized "
            "device kernel time only — excludes host<->device staging and pipeline (re)build "
            "cost; see fzgpumodules:rebuilt to check whether the latter applies to this call.");
        set(options, "fzgpumodules:rebuilt",
            "[Output] True if the last compress() call (re)built the pipeline (first call, "
            "dirty options, or — in manual-pipeline mode — a changed input size) rather than "
            "reusing the cached one. execution_time_us excludes build cost either way, so a "
            "timing loop should discard measurements where this is true, or warm up first.");

        if (config_file_.empty()) {
            for_each_stage([&](StageKind& sk, const std::string& sid, const std::string& tok) {
                sk.populate_documentation(options, sid, tok);
                if(!sk.outlier_indices_key(tok).empty())
                    set(options, "fzgpumodules:" + sid + ":outlier_count",
                        std::string("[Output] Actual outlier count from last compress"));
            });
        }

        return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
        // Serialized callers see either the complete update or the unchanged configuration.
        auto fields = std::tie(error_bound_abs_, error_bound_rel_, stages_, connections_,
            primary_source_, memory_strategy_, memory_multiplier_, error_bound_mode_,
            dims_, num_streams_, graph_mode_, specialization_, config_file_, expose_stage_outputs_);
        auto saved = std::apply([](const auto&... value) { return std::make_tuple(value...); }, fields);
        auto saved_fields = std::apply([](auto&... value) { return std::tie(value...); }, saved);
        std::vector<std::unique_ptr<StageKind>> saved_kinds;
        for(const auto& kind : stage_kinds_) saved_kinds.push_back(kind->clone());
        const bool saved_dirty = state_.dirty;
        auto rollback = [&]() noexcept {
            fields.swap(saved_fields);
            stage_kinds_.swap(saved_kinds);
            state_.dirty = saved_dirty;
        };
        try {
            const int status = apply_options(options);
            if(status != 0) rollback();
            return status;
        } catch(const std::exception& error) {
            rollback();
            return set_error(1, std::string("Invalid FZGM configuration: ") + error.what());
        } catch(...) {
            rollback();
            throw;
        }
    }

    template<typename T>
    auto read_option(const pressio_options& options, const std::string& key, T* value) {
        const auto status = get(options, key, value);
        if(status == pressio_options_key_exists)
            throw std::invalid_argument(key + " has the wrong type or no value");
        return status;
    }

    int apply_options(struct pressio_options const& options) {
        double old_abs = error_bound_abs_, old_rel = error_bound_rel_;
        read_option(options, "pressio:abs", &error_bound_abs_);
        read_option(options, "pressio:rel", &error_bound_rel_);

        auto old_stages = stages_, old_conns = connections_;
        auto old_primary_source = primary_source_;
        read_option(options, "fzgpumodules:stages",         &stages_);
        read_option(options, "fzgpumodules:connections",    &connections_);
        read_option(options, "fzgpumodules:primary_source", &primary_source_);

        std::string old_strategy = memory_strategy_;
        read_option(options, "fzgpumodules:memory_strategy", &memory_strategy_);
        if(!MEMORY_STRATEGIES.contains(memory_strategy_)) {
            auto invalid = memory_strategy_;
            return set_error(1, "Invalid fzgpumodules:memory_strategy: '" + invalid +
                                "'. Valid: preallocate, minimal");
        }

        double old_mult = memory_multiplier_;
        read_option(options, "fzgpumodules:memory_multiplier", &memory_multiplier_);

        std::string old_mode = error_bound_mode_;
        read_option(options, "fzgpumodules:error_bound_mode", &error_bound_mode_);
        if(!ERROR_BOUND_MODES.contains(error_bound_mode_)) {
            auto invalid = error_bound_mode_;
            return set_error(1, "Invalid fzgpumodules:error_bound_mode: '" + invalid +
                                "'. Valid: abs, rel, noa, prel");
        }

        auto old_dims = dims_;
        pressio_data dims_data;
        if(read_option(options, "fzgpumodules:dims", &dims_data) == pressio_options_key_set) {
            std::vector<size_t> v;
            switch(dims_data.dtype()) {
                case pressio_uint8_dtype: case pressio_uint16_dtype:
                case pressio_uint32_dtype: case pressio_uint64_dtype:
                    for(auto dim : dims_data.to_vector<uint64_t>()) {
                        if(dim > std::numeric_limits<size_t>::max())
                            throw std::invalid_argument("dims exceed the host size range");
                        v.push_back(static_cast<size_t>(dim));
                    }
                    break;
                case pressio_int8_dtype: case pressio_int16_dtype:
                case pressio_int32_dtype: case pressio_int64_dtype:
                    for(auto dim : dims_data.to_vector<int64_t>()) {
                        if(dim < 0 || uint64_t(dim) > std::numeric_limits<size_t>::max())
                            throw std::invalid_argument("dims must be nonnegative host sizes");
                        v.push_back(static_cast<size_t>(dim));
                    }
                    break;
                default: throw std::invalid_argument("dims must contain integer values");
            }
            if(v.empty() || v.size() > 3)
                throw std::invalid_argument("dims must contain one to three values");
            dims_.assign(3, size_t(1));
            for(size_t i = 0; i < v.size(); ++i) {
                if(i > 0 && v[i] == 0)
                    throw std::invalid_argument("only the first dimension may be zero (automatic)");
                dims_[i] = v[i];
            }
        }

        int64_t old_streams = num_streams_;
        read_option(options, "fzgpumodules:num_streams", &num_streams_);

        bool old_graph = graph_mode_;
        read_option(options, "fzgpumodules:graph_mode", &graph_mode_);

        std::string removed_fusion;
        if (get(options, "fzgpumodules:fusion", &removed_fusion) != pressio_options_key_does_not_exist)
            return set_error(1, "fzgpumodules:fusion was removed; use fzgpumodules:specialization");
        const std::string old_specialization = specialization_;
        std::string requested_specialization = specialization_;
        auto status = get(options, "fzgpumodules:specialization", &requested_specialization);
        if (status == pressio_options_key_exists)
            return set_error(1, "fzgpumodules:specialization must be a string");
        if (!SPECIALIZATION_POLICIES.contains(requested_specialization))
            return set_error(1, "Invalid fzgpumodules:specialization: '" + requested_specialization +
                                "'. Valid: off, auto");
        specialization_ = requested_specialization;

        std::string old_config_file = config_file_;
        read_option(options, "fzgpumodules:config_file", &config_file_);

        bool old_expose = expose_stage_outputs_;
        read_option(options, "fzgpumodules:expose_stage_outputs", &expose_stage_outputs_);

        if(!std::isfinite(error_bound_abs_) || error_bound_abs_ < 0 ||
           !std::isfinite(error_bound_rel_) || error_bound_rel_ < 0)
            throw std::invalid_argument("error bounds must be finite and nonnegative");
        if(!std::isfinite(memory_multiplier_) || memory_multiplier_ <= 0 ||
           memory_multiplier_ > std::numeric_limits<float>::max() ||
           static_cast<float>(memory_multiplier_) == 0)
            throw std::invalid_argument("memory_multiplier must be a finite positive float");
        if(num_streams_ < 1 || num_streams_ > std::numeric_limits<int>::max())
            throw std::invalid_argument("num_streams must fit a positive int");

        bool stage_params_changed = false;
        for_each_stage([&](StageKind& sk, const std::string& sid, const std::string& tok) {
            pressio_options expected;
            sk.populate_options(expected, sid, tok);
            for(const auto& entry : expected) {
                if(options.key_status(entry.first) == pressio_options_key_exists ||
                   (options.key_status(entry.first) == pressio_options_key_set &&
                    options.get(entry.first).type() != entry.second.type()))
                    throw std::invalid_argument(entry.first + " has the wrong type");
            }
            stage_params_changed |= sk.read_options(options, sid, tok);
        });

        if(stages_ != old_stages || connections_ != old_conns ||
           primary_source_ != old_primary_source ||
           memory_strategy_ != old_strategy || memory_multiplier_ != old_mult ||
           error_bound_abs_ != old_abs || error_bound_rel_ != old_rel ||
           error_bound_mode_ != old_mode || dims_ != old_dims ||
           num_streams_ != old_streams || graph_mode_ != old_graph ||
           specialization_ != old_specialization ||
           config_file_ != old_config_file || expose_stage_outputs_ != old_expose ||
           stage_params_changed)
        {
            state_.dirty = true;
        }

        return 0;
    }

    // =========================================================================
    // Compression / decompression
    // =========================================================================

    int compress_impl(const pressio_data* input, struct pressio_data* output) override {
        try {
            switch(input->dtype()) {
                case pressio_float_dtype:   return compress_device<float>(input, output);
                case pressio_double_dtype:  return compress_device<double>(input, output);
                case pressio_int8_dtype:    return compress_device<int8_t>(input, output);
                case pressio_int16_dtype:   return compress_device<int16_t>(input, output);
                case pressio_int32_dtype:   return compress_device<int32_t>(input, output);
                case pressio_int64_dtype:   return compress_device<int64_t>(input, output);
                case pressio_uint8_dtype:   return compress_device<uint8_t>(input, output);
                case pressio_uint16_dtype:  return compress_device<uint16_t>(input, output);
                case pressio_uint32_dtype:  return compress_device<uint32_t>(input, output);
                case pressio_uint64_dtype:  return compress_device<uint64_t>(input, output);
                default: return set_error(1, "Unsupported data type for FZGpuModules compression");
            }
        } catch(std::exception const& ex) {
            return set_error(2, std::string("Compression exception: ") + ex.what());
        }
    }

    int decompress_impl(const pressio_data* input, struct pressio_data* output) override {
        try {
            return decompress_archive(input, output);
        } catch(std::exception const& ex) {
            return set_error(2, std::string("Decompression exception: ") + ex.what());
        }
    }

    // =========================================================================
    // Metadata
    // =========================================================================

    int major_version() const override { return 0; }
    int minor_version() const override { return 1; }
    int patch_version() const override { return 0; }
    const char* version() const override { return "0.1.0"; }
    const char* prefix() const override { return "fzgpumodules"; }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
        auto p = compat::make_unique<fzgpumodules_plugin>();
        p->set_options(get_options());
        return p;
    }

    pressio_options get_metrics_results_impl() const override {
        pressio_options metrics;
        set(metrics, "fzgpumodules:peak_memory",       last_peak_memory_);
        set(metrics, "fzgpumodules:execution_time_us", last_execution_time_us_);
        set(metrics, "fzgpumodules:rebuilt",           last_rebuilt_);
        // Inverse decisions become available after the first decode.
        uint64_t forward_groups = 0, inverse_groups = 0;
        std::vector<std::string> forward_impls, inverse_impls;
        std::string fallback = "pipeline_not_built";
        if (state_.ptr) {
            const auto& info = state_.ptr->getSpecializationInfo();
            forward_groups = info.installed_groups.size();
            inverse_groups = info.installed_inverse_groups.size();
            for (const auto& group : info.installed_groups)
                forward_impls.push_back(group.implementation);
            for (const auto& group : info.installed_inverse_groups)
                inverse_impls.push_back(group.implementation);
            fallback = info.fallback_reason;
        }
        set(metrics, "fzgpumodules:specialization_forward_groups", forward_groups);
        set(metrics, "fzgpumodules:specialization_inverse_groups", inverse_groups);
        set(metrics, "fzgpumodules:specialization_forward_implementations", forward_impls);
        set(metrics, "fzgpumodules:specialization_inverse_implementations", inverse_impls);
        set(metrics, "fzgpumodules:specialization_fallback_reason", fallback);
        for(auto& [sid, count] : last_outlier_counts_)
            set(metrics, "fzgpumodules:" + sid + ":outlier_count", count);
        for(auto& [key, val] : last_stage_outputs_)
            set(metrics, key, val);
        return metrics;
    }

private:
    // =========================================================================
    // Stage-aware pipeline builder
    // =========================================================================

    void build_pipeline(size_t data_size) {
        if (graph_mode_) {
            for (auto& tok : stages_)
                if (token_kind(tok) == "rze")
                    throw std::runtime_error(
                        "fzgpumodules:graph_mode=true is incompatible with the rze stage");
        }

        fz::MemoryStrategy strategy = graph_mode_
            ? fz::MemoryStrategy::PREALLOCATE
            : MEMORY_STRATEGIES.at(memory_strategy_);

        state_.outlier_stages.clear();
        state_.ptr.reset(new fz::Pipeline(data_size, strategy, static_cast<float>(memory_multiplier_)));
        const std::array<size_t,3> fz_dims = {dims_[0], dims_[1], dims_[2]};
        configure_pipeline_base(*state_.ptr, &fz_dims, static_cast<int>(num_streams_), SPECIALIZATION_POLICIES.at(specialization_));

        StageContext ctx{
            *state_.ptr,
            ERROR_BOUND_MODES.at(error_bound_mode_),
            (error_bound_mode_ == "rel") ? error_bound_rel_ : error_bound_abs_
        };

        std::map<std::string, fz::Stage*> ptrs;
        for(size_t i = 0; i < stages_.size(); i++) {
            std::string sid = "s" + std::to_string(i);
            auto kv = token_kind(stages_[i]);
            fz::Stage* stage = nullptr;
            for(auto& sk : stage_kinds_) {
                if(sk->matches(kv)) {
                    stage = sk->add_stage(stages_[i], sid, ctx);
                    std::string okey = sk->outlier_indices_key(stages_[i]);
                    if(!okey.empty())
                        state_.outlier_stages.push_back({sid, {stage, okey}});
                    break;
                }
            }
            if(!stage)
                throw std::runtime_error("Unknown stage type: " + stages_[i]);
            ptrs[sid] = stage;
        }

        for(auto& conn_str : connections_) {
            auto c = parse_connection(conn_str);
            if(c.from == "__external__") {
                // Bind the pipeline's raw input directly to this port, even
                // though 'to' may also have other real connections (e.g.
                // cdf97_outlier_correct). Port position follows the same
                // connection-order convention as every other wiring string
                // here -- list "sN <- __external__" before any "sN <- sM"
                // entry that should land on a later port of the same stage.
                state_.ptr->bindExternalInput(ptrs.at(c.to));
                continue;
            }
            if(c.port.empty())
                state_.ptr->connect(ptrs.at(c.to), ptrs.at(c.from));
            else
                state_.ptr->connect(ptrs.at(c.to), ptrs.at(c.from), c.port);
        }

        if(!primary_source_.empty())
            state_.ptr->setPrimarySource(ptrs.at(primary_source_));

        // Build ordered list of terminal (unconnected) output ports for expose_stage_outputs.
        // A port is terminal if no connection names it as the 'from' source.
        // An empty port in a connection means the stage's first (index-0) port.
        state_.terminal_outputs.clear();
        if (expose_stage_outputs_) {
            auto is_consumed = [&](const std::string& sid, const std::string& port, size_t idx) {
                for (auto& conn_str : connections_) {
                    auto c = parse_connection(conn_str);
                    if (c.from != sid) continue;
                    if (c.port == port)              return true;  // explicit match
                    if (c.port.empty() && idx == 0)  return true;  // default = first port
                }
                return false;
            };
            for (size_t i = 0; i < stages_.size(); i++) {
                std::string sid = "s" + std::to_string(i);
                fz::Stage*  stage = ptrs[sid];
                auto names = stage->getOutputNames();
                if (names.empty()) names = {"output"};
                for (size_t j = 0; j < names.size(); j++) {
                    if (!is_consumed(sid, names[j], j)) {
                        state_.terminal_outputs.push_back({
                            "fzgpumodules:" + sid + ":output:" + names[j],
                            fz_to_pressio_dtype(static_cast<fz::DataType>(stage->getOutputDataType(j)))
                        });
                    }
                }
            }
        }

        if (graph_mode_)
            state_.ptr->enableGraphMode(true);

        state_.ptr->finalize();
        maybe_capture_graph(*state_.ptr, stream_, graph_mode_);
    }

    void build_from_config() {
        fz::MemoryStrategy strategy = graph_mode_
            ? fz::MemoryStrategy::PREALLOCATE
            : MEMORY_STRATEGIES.at(memory_strategy_);

        state_.outlier_stages.clear();
        state_.ptr.reset(new fz::Pipeline(0, strategy, static_cast<float>(memory_multiplier_)));
        configure_pipeline_base(*state_.ptr, nullptr, static_cast<int>(num_streams_), SPECIALIZATION_POLICIES.at(specialization_));

        if (graph_mode_)
            state_.ptr->enableGraphMode(true);

        state_.ptr->loadConfig(config_file_);  // adds stages, wires, and finalizes
        maybe_capture_graph(*state_.ptr, stream_, graph_mode_);
    }

    // =========================================================================
    // Typed compress / decompress
    // =========================================================================

    template<typename T>
    int compress_device(const pressio_data* real_input, struct pressio_data* output) {
        pressio_data gpu_input = domain_manager().make_readable(
            domain_plugins().build("cudamalloc"), *real_input);
        const T* d_input   = static_cast<const T*>(gpu_input.data());
        size_t   data_size = gpu_input.size_in_bytes();

        // Config-file pipelines are sized by the TOML's input_size; don't rebuild
        // on data_size change. Manual pipelines must rebuild when data size changes.
        bool need_rebuild = state_.dirty || !state_.ptr ||
                            (config_file_.empty() && data_size != state_.last_size);
        last_rebuilt_ = need_rebuild;
        if(need_rebuild) {
            try {
                if (!config_file_.empty())
                    build_from_config();
                else
                    build_pipeline(data_size);
                state_.dirty     = false;
                state_.last_size = data_size;
            } catch(std::exception const& ex) {
                return set_error(3, std::string("Pipeline construction failed: ") + ex.what());
            }
        }

        // compress() returns a BorrowedDeviceBuffer — pool-owned, valid until the
        // next compress()/reset(). We copy it out below into a pressio-owned buffer.
        fz::BorrowedDeviceBuffer comp;

        auto start = std::chrono::high_resolution_clock::now();
        try {
            comp = state_.ptr->compress({d_input, data_size}, stream_);
        } catch(std::exception const& ex) {
            return set_error(4, std::string("Compression execution failed: ") + ex.what());
        }
        const size_t output_size = comp.bytes();
        auto end = std::chrono::high_resolution_clock::now();
        last_execution_time_us_ =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        last_peak_memory_ = state_.ptr->getPeakMemoryUsage();

        last_outlier_counts_.clear();
        for(auto& [sid, ref] : state_.outlier_stages) {
            auto sizes = ref.ptr->getActualOutputSizesByName();
            auto it    = sizes.find(ref.indices_key);
            last_outlier_counts_[sid] = (it != sizes.end())
                ? static_cast<int64_t>(it->second / sizeof(uint32_t)) : 0;
        }

        last_stage_outputs_.clear();
        if (expose_stage_outputs_ && !state_.terminal_outputs.empty()) {
            std::vector<uint8_t> h_buf(output_size);
            cudaMemcpy(h_buf.data(), comp.data(), output_size, cudaMemcpyDeviceToHost);
            parse_stage_outputs(h_buf.data(), output_size);
        }

        auto metadata = make_archive_metadata(*real_input,
            state_.ptr->serializeHeaderToMemory(), output_size);
        const size_t archive_size = metadata.size() + output_size;
        auto archive = pressio_data::owning(pressio_byte_dtype, {archive_size},
                                           domain_plugins().build("cudamalloc"));
        auto check_cuda = [](cudaError_t error) {
            if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        };
        check_cuda(cudaMemcpyAsync(archive.data(), metadata.data(), metadata.size(),
                                   cudaMemcpyHostToDevice, stream_));
        check_cuda(cudaMemcpyAsync(static_cast<uint8_t*>(archive.data()) + metadata.size(),
                                   comp.data(), output_size, cudaMemcpyDeviceToDevice, stream_));
        check_cuda(cudaStreamSynchronize(stream_));
        const bool wants_device = real_input->domain()->domain_id() == "cudamalloc" ||
                                  output->domain()->domain_id() == "cudamalloc";
        *output = wants_device ? std::move(archive)
            : domain_manager().make_readable(domain_plugins().build("malloc"), std::move(archive));
        return 0;
    }

    int decompress_archive(const pressio_data* input, pressio_data* output) {
        const bool wants_device = output->domain()->domain_id() == "cudamalloc";
        auto compressed = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *input);
        const size_t total = compressed.size_in_bytes();
        if(total < ArchiveMetadata::prefix_size)
            throw std::runtime_error("Truncated FZGM archive prefix");
        auto check_cuda = [](cudaError_t error) {
            if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        };
        std::vector<uint8_t> metadata(ArchiveMetadata::prefix_size);
        check_cuda(cudaMemcpy(metadata.data(), compressed.data(), metadata.size(), cudaMemcpyDeviceToHost));
        const size_t metadata_size = archive_metadata_size(metadata.data(), total);
        metadata.resize(metadata_size);
        check_cuda(cudaMemcpy(metadata.data(), compressed.data(), metadata.size(), cudaMemcpyDeviceToHost));
        const auto archive = parse_archive_metadata(metadata.data(), total);
        if(output->num_elements() != 0 &&
           (output->dtype() != archive.dtype || output->dimensions() != archive.dims))
            throw std::runtime_error("FZGM archive dtype/shape does not match requested output");

        if(archive.segment_sizes.size() > 1) {
            const size_t concat_size = 4 + 8*archive.segment_sizes.size();
            if(concat_size > archive.payload_size)
                throw std::runtime_error("Truncated FZGM concat header");
            std::vector<uint8_t> concat(concat_size);
            check_cuda(cudaMemcpy(concat.data(), static_cast<uint8_t*>(compressed.data()) +
                                   archive.payload_offset, concat.size(), cudaMemcpyDeviceToHost));
            if(archive_read(concat.data(), 4) != archive.segment_sizes.size())
                throw std::runtime_error("FZGM concat count disagrees with archive metadata");
            for(size_t i = 0; i < archive.segment_sizes.size(); ++i)
                if(archive_read(concat.data()+4+8*i, 8) != archive.segment_sizes[i])
                    throw std::runtime_error("FZGM concat length disagrees with archive metadata");
        }
        // Reconstruct from this payload, never from the encoder's most recent state.
        auto decoder = fz::Pipeline::fromHeader(archive.header.data(), archive.header.size(),
            MEMORY_STRATEGIES.at(memory_strategy_), SPECIALIZATION_POLICIES.at(specialization_));
        auto restored = decoder->decompressOwned(
            {static_cast<uint8_t*>(compressed.data()) + archive.payload_offset, archive.payload_size}, stream_);
        if(restored.bytes() != archive.output_size)
            throw std::runtime_error("FZGM decoded size does not match archive shape");
        auto gpu_output = pressio_data::move(archive.dtype, restored.release(), archive.dims,
                                            domain_plugins().build("cudamalloc"));
        *output = wants_device ? std::move(gpu_output)
            : domain_manager().make_readable(domain_plugins().build("malloc"), std::move(gpu_output));
        state_.ptr = std::move(decoder);
        state_.dirty = true; // A later encode must reconstruct its configured forward pipeline.
        state_.outlier_stages.clear();
        state_.terminal_outputs.clear();
        return 0;
    }

    // =========================================================================
    // Dispatch helpers: call f(handler, sid, token) for each stage
    // =========================================================================

    template<typename F>
    void for_each_stage(F&& f) const {
        for(size_t i = 0; i < stages_.size(); i++) {
            std::string sid = "s" + std::to_string(i);
            auto kv = token_kind(stages_[i]);
            for(const auto& sk : stage_kinds_)
                if(sk->matches(kv)) { f(*sk, sid, stages_[i]); break; }
        }
    }

    template<typename F>
    void for_each_stage(F&& f) {
        for(size_t i = 0; i < stages_.size(); i++) {
            std::string sid = "s" + std::to_string(i);
            auto kv = token_kind(stages_[i]);
            for(auto& sk : stage_kinds_)
                if(sk->matches(kv)) { f(*sk, sid, stages_[i]); break; }
        }
    }

    // =========================================================================
    // parse_stage_outputs — maps the compressed blob to per-port pressio_data.
    //
    // Single terminal buffer: pipeline omits the concat header and returns raw
    // stage output directly.  Multi-buffer: concat format from writeConcatBuffer():
    //   [n_bufs: u32][sz_0: u64]...[sz_N-1: u64]  ← padded to 16-byte boundary
    //   [data_0: align16(sz_0) bytes] [data_1: ...] ...
    // =========================================================================

    void parse_stage_outputs(const uint8_t* buf, size_t total) {
        last_stage_outputs_.clear();
        if (state_.terminal_outputs.empty()) return;

        if (state_.terminal_outputs.size() == 1) {
            auto& t = state_.terminal_outputs[0];
            size_t esz = static_cast<size_t>(pressio_dtype_size(t.dtype));
            if (esz == 0) esz = 1;
            last_stage_outputs_[t.metric_key] =
                pressio_data::copy(t.dtype, buf, {total / esz});
            return;
        }

        uint32_t n_bufs = 0;
        std::memcpy(&n_bufs, buf, sizeof(uint32_t));

        std::vector<uint64_t> sizes(n_bufs);
        const uint8_t* hdr = buf + sizeof(uint32_t);
        for (uint32_t i = 0; i < n_bufs; i++) {
            std::memcpy(&sizes[i], hdr, sizeof(uint64_t));
            hdr += sizeof(uint64_t);
        }

        size_t hdr_raw    = sizeof(uint32_t) + n_bufs * sizeof(uint64_t);
        size_t hdr_padded = (hdr_raw + 15u) & ~15u;
        const uint8_t* data_ptr = buf + hdr_padded;

        for (uint32_t i = 0; i < n_bufs && i < static_cast<uint32_t>(state_.terminal_outputs.size()); i++) {
            uint64_t sz = sizes[i];
            auto& t = state_.terminal_outputs[i];
            size_t esz = static_cast<size_t>(pressio_dtype_size(t.dtype));
            if (esz == 0) esz = 1;
            if (sz > 0) {
                last_stage_outputs_[t.metric_key] =
                    pressio_data::copy(t.dtype, data_ptr, {sz / esz});
            } else {
                last_stage_outputs_[t.metric_key] =
                    pressio_data::owning(t.dtype, {0});
            }
            data_ptr += (sz + 15u) & ~15u;
        }
    }

    // =========================================================================
    // Member variables
    // =========================================================================

    CudaStream    stream_;
    PipelineState state_;

    std::vector<std::unique_ptr<StageKind>> stage_kinds_;

    double      error_bound_abs_   = 1e-3;
    double      error_bound_rel_   = 1e-3;
    std::string error_bound_mode_  = "abs";

    std::string memory_strategy_   = "minimal";
    // double/int64_t (not float/int): libpressio's Python layer boxes every
    // Python float as C++ double and every Python int as C++ int64_t, and
    // pressio_options::get() requires an exact type match — no numeric
    // coercion — so a narrower field here would silently never be set.
    double      memory_multiplier_ = 3.0;
    std::string specialization_            = "off";
    std::vector<size_t> dims_      = {0, 1, 1};
    int64_t     num_streams_       = 1;

    std::vector<std::string> stages_        = {"lorenzo:float:uint16", "diff:uint16"};
    std::vector<std::string> connections_   = {"s1 <- s0:codes"};
    // "" = default (Pipeline::setPrimarySource() not called; the sole/
    // first-discovered source is used). Set to a stage id ("s2") to pick a
    // specific stage's inverse output as decompress()'s answer -- needed for
    // pipelines with more than one source stage, e.g. the SPERR recipe
    // (docs/libpressio_python.md) where "s2 <- __external__" alongside
    // "s2 <- s1:codes" makes cdf97_outlier_correct a second source.
    std::string               primary_source_ = "";
    std::string              config_file_   = "";

    bool    graph_mode_             = false;
    bool    expose_stage_outputs_   = false;

    size_t  last_peak_memory_       = 0;
    int64_t last_execution_time_us_ = 0;
    bool    last_rebuilt_           = false;  // did the last compress() rebuild the pipeline?
    std::map<std::string, int64_t>    last_outlier_counts_;
    std::map<std::string, pressio_data> last_stage_outputs_;
};

// =============================================================================
// Plugin registration
// =============================================================================

pressio_register registration(
    compressor_plugins(),
    "fzgpumodules",
    []() { return compat::make_unique<fzgpumodules_plugin>(); }
);

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
