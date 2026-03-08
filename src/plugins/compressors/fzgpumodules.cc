#include <vector>
#include <memory>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include <compare>
#include <format>
#include <ranges>
#include <regex>
#include "std_compat/memory.h"
#include "std_compat/utility.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

#include "fzgpumodules.h"

namespace libpressio { namespace compressors { namespace fzgpumodules_ns {

// =============================================================================
// Helper Functions & Constants
// =============================================================================

// Memory strategy mapping
static const std::map<std::string, fz::MemoryStrategy> MEMORY_STRATEGIES {
    {"preallocate", fz::MemoryStrategy::PREALLOCATE},
    {"pipeline", fz::MemoryStrategy::PIPELINE},
    {"minimal", fz::MemoryStrategy::MINIMAL}, 
};

// Helper to extract keys from maps
template <class T>
std::vector<std::string> map_keys(std::map<std::string, T> const& map) {
    std::vector<std::string> keys;
    keys.reserve(map.size());
    for (auto const& i : map) {
        keys.push_back(i.first);
    }
    return keys;
}

// =============================================================================
// Pipeline Builder Helpers
// =============================================================================

static std::vector<std::string> split_str(std::string_view s, char delim) {
    std::vector<std::string> parts;
    for (auto part : s | std::views::split(delim))
        parts.emplace_back(part.begin(), part.end());
    return parts;
}

// Parsed connection from "sN <- sM:port" or "sN <- sM"
struct ParsedConnection {
    std::string to, from, port;
};

static ParsedConnection parse_connection(const std::string& s) {
    static const std::regex re(R"(^\s*(\S+)\s*<-\s*(\S+?)(?::(\S+))?\s*$)");
    std::smatch m;
    if (!std::regex_match(s, m, re))
        throw std::invalid_argument("Invalid connection (expected 'sN <- sM[:port]'): " + s);
    return {m[1].str(), m[2].str(), m[3].str()};
}

// =============================================================================
// RAII Helpers
// =============================================================================

// Wraps cudaStream_t
struct CudaStream {
    cudaStream_t s = nullptr;
    CudaStream()  { cudaStreamCreate(&s); }
    ~CudaStream() { if(s) cudaStreamDestroy(s); }
    CudaStream(CudaStream&& o) noexcept : s(std::exchange(o.s, nullptr)) {}
    CudaStream& operator=(CudaStream&& o) noexcept {
        if(this != &o) { if(s) cudaStreamDestroy(s); s = std::exchange(o.s, nullptr); }
        return *this;
    }
    CudaStream(CudaStream const&) { cudaStreamCreate(&s); }
    CudaStream& operator=(CudaStream const&) { if(!s) cudaStreamCreate(&s); return *this; }
    operator cudaStream_t() const { return s; }
};

// Wraps pipeline state
struct PipelineState {
    std::unique_ptr<fz::Pipeline> ptr;
    bool dirty = true;
    size_t last_size = 0;

    PipelineState() = default;
    PipelineState(PipelineState&&) = default;
    PipelineState& operator=(PipelineState&&) = default;
    PipelineState(PipelineState const&) : ptr(nullptr), dirty(true), last_size(0) {}
    PipelineState& operator=(PipelineState const&) { ptr.reset(); dirty = true; last_size = 0; return *this; }
};

// =============================================================================
// Main Plugin Class
// =============================================================================

class fzgpumodules_plugin : public libpressio_compressor_plugin {
public:
    
    fzgpumodules_plugin() {
        // Seed default Lorenzo params for s0
        lorenzo_params_["s0"] = LorenzoParams{};
    }

    // =========================================================================
    // Options Interface
    // =========================================================================
    
    struct pressio_options get_options_impl() const override {
        struct pressio_options options;
        
        // TODO: relative error bound mode
        set(options, "pressio:abs", error_bound_);
        
        // Global pipeline settings
        set(options, "fzgpumodules:memory_strategy", memory_strategy_);
        set(options, "fzgpumodules:memory_multiplier", memory_multiplier_);
        
        // Pipeline topology
        set(options, "fzgpumodules:stages", stages_);
        set(options, "fzgpumodules:connections", connections_);
        
        // Per-stage parameters
        populate_lorenzo_options(options);
        
        return options;
    }
    
    struct pressio_options get_configuration_impl() const override {
        struct pressio_options options;
        
        set(options, "pressio:thread_safe", pressio_thread_safety_serialized);
        set(options, "fzgpumodules:memory_strategy", map_keys(MEMORY_STRATEGIES));
        
        return options;
    }
    
    struct pressio_options get_documentation_impl() const override {
        struct pressio_options options;
        
        set(options, "pressio:description",
            "FZGpuModules GPU compressor with user-composable pipeline stages.\n\n"
            "Configure the pipeline by setting `fzgpumodules:stages` (ordered list of stage tokens) "
            "and `fzgpumodules:connections` (list of wiring strings). "
            "Per-stage Lorenzo parameters are exposed as `fzgpumodules:sN:quant_radius` and "
            "`fzgpumodules:sN:outlier_capacity` for each Lorenzo stage sN.");
        
        set(options, "fzgpumodules:memory_strategy",
            "GPU memory allocation strategy:\n"
            "  preallocate: allocate everything upfront for speed, higher peak memory\n"
            "  pipeline: preallocate critical path, free stages when done\n"
            "  minimal: allocate only what each stage needs at execution time");
        set(options, "fzgpumodules:memory_multiplier",
            "Memory allocation multiplier (e.g., 3.0 = 3x input size)");
        set(options, "fzgpumodules:stages",
            "Ordered list of stage tokens defining the pipeline stages.\n"
            "Supported tokens: lorenzo:float:uint16, lorenzo:double:uint16, "
            "diff:uint16, diff:uint32, diff:float, rle:uint16, rle:uint32, passthrough, scale.\n"
            "Example: {\"lorenzo:float:uint16\", \"diff:uint16\"}");
        set(options, "fzgpumodules:connections",
            "List of wiring strings between stages in the form 'sN <- sM' or 'sN <- sM:port'.\n"
            "Example: {\"s1 <- s0:codes\"} connects s1 to the 'codes' output port of s0.");
        populate_lorenzo_documentation(options);
        set(options, "fzgpumodules:peak_memory",
            "[Output] Peak GPU memory usage from last compression (bytes)");
        set(options, "fzgpumodules:execution_time_us",
            "[Output] Execution time from last compression (microseconds)");
        
        return options;
    }
    
    int set_options_impl(struct pressio_options const& options) override {
        // Standard error bound
        double old_error_bound = error_bound_;
        get(options, "pressio:abs", &error_bound_);
        
        // Topology
        auto old_stages = stages_;
        auto old_connections = connections_;
        get(options, "fzgpumodules:stages", &stages_);
        get(options, "fzgpumodules:connections", &connections_);
        
        // Global settings
        std::string old_strategy = memory_strategy_;
        float old_multiplier = memory_multiplier_;
        get(options, "fzgpumodules:memory_strategy", &memory_strategy_);
        if (MEMORY_STRATEGIES.find(memory_strategy_) == MEMORY_STRATEGIES.end()) {
            auto invalid = memory_strategy_;
            memory_strategy_ = old_strategy;
            return set_error(1, "Invalid fzgpumodules:memory_strategy: '" + invalid + "'. Valid values are: preallocate, pipeline, minimal");
        }
        get(options, "fzgpumodules:memory_multiplier", &memory_multiplier_);
        
        // Per-stage Lorenzo params
        auto old_params = lorenzo_params_;
        read_lorenzo_options(options);
        
        if(stages_ != old_stages || connections_ != old_connections ||
           memory_strategy_ != old_strategy ||
           memory_multiplier_ != old_multiplier ||
           error_bound_ != old_error_bound ||
           lorenzo_params_ != old_params) {
            state_.dirty = true;
        }
        
        return 0;
    }

    // =========================================================================
    // Compression/Decompression
    // =========================================================================
    
    int compress_impl(const pressio_data* input, struct pressio_data* output) override {
        try {
            // Dispatch based on data type
            switch(input->dtype()) {
                case pressio_float_dtype:
                    return compress_device<float>(input, output);
                case pressio_double_dtype:
                    return compress_device<double>(input, output);
                default:
                    return set_error(1, "Unsupported data type for FZGpuModules compression");
            }
        } catch (std::exception const& ex) {
            return set_error(2, std::string("Compression exception: ") + ex.what());
        }
    }
    
    int decompress_impl(const pressio_data* input, struct pressio_data* output) override {
        try {
            // Dispatch based on output data type
            switch(output->dtype()) {
                case pressio_float_dtype:
                    return decompress_typed<float>(input, output);
                case pressio_double_dtype:
                    return decompress_typed<double>(input, output);
                default:
                    return set_error(1, "Unsupported data type for FZGpuModules decompression");
            }
        } catch (std::exception const& ex) {
            return set_error(2, std::string("Decompression exception: ") + ex.what());
        }
    }

    // =========================================================================
    // Typed Compression/Decompression Templates
    // =========================================================================

    template<typename T>
    int compress_device(const pressio_data* real_input, struct pressio_data* output) {
        // H2D
        pressio_data gpu_input = domain_manager().make_readable(
            domain_plugins().build("cudamalloc"), *real_input);
        const T* d_input = static_cast<const T*>(gpu_input.data());
        size_t data_size = gpu_input.size_in_bytes();
        
        // Build pipeline if needed (also rebuild if input size has changed)
        if(state_.dirty || !state_.ptr || data_size != state_.last_size) {
            try {
                build_pipeline(data_size);
                state_.dirty = false;
                state_.last_size = data_size;
            } catch (std::exception const& ex) {
                return set_error(3, std::string("Pipeline construction failed: ") + ex.what());
            }
        }
        
        // Run compression with timing
        void* d_output = nullptr;
        size_t output_size = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            state_.ptr->compress(d_input, data_size, &d_output, &output_size, stream_);
        } catch (std::exception const& ex) {
            return set_error(4, std::string("Compression execution failed: ") + ex.what());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        last_execution_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        last_peak_memory_ = state_.ptr->getPeakMemoryUsage();
        
        // Wrap GPU output
        *output = pressio_data::move(
            pressio_byte_dtype, d_output, {output_size},
            domain_plugins().build("cudamalloc"));
        
        return 0;
    }
    
    template<typename T>
    int decompress_typed(const pressio_data* compressed_input, struct pressio_data* output) {
        if(!state_.ptr) {
            return set_error(5, "Decompression requires a pipeline built by a prior compress call");
        }
        const size_t original_size = output->size_in_bytes();
        // Save output metadata before we overwrite
        const pressio_dtype out_dtype = output->dtype();
        const auto out_dims = output->dimensions();
        
        // H2D 
        pressio_data gpu_compressed = domain_manager().make_readable(
            domain_plugins().build("cudamalloc"), *compressed_input);
        const void* d_compressed = gpu_compressed.data();
        
        void* d_output = nullptr;
        size_t output_size = 0;
        
        try {
            // decompress() runs on same pipeline as compress
            state_.ptr->decompress(d_compressed, original_size, &d_output, &output_size, stream_);
        } catch(std::exception const& ex) {
            return set_error(4, std::string("Decompression execution failed: ") + ex.what());
        }
        
        // Wrap GPU output
        pressio_data gpu_output = pressio_data::move(
            out_dtype, d_output, out_dims,
            domain_plugins().build("cudamalloc"));
        *output = domain_manager().make_readable(
            domain_plugins().build("malloc"), std::move(gpu_output));
        
        return 0;
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    // Create a stage from a token (e.g. "lorenzo:float:uint16"), add to pipeline, return observer ptr
    fz::Stage* add_stage_from_token(const std::string& token, const std::string& sid) {
        auto parts = split_str(token, ':');
        const std::string& kind = parts[0];
        
        if(kind == "lorenzo") {
            // Defaults: float input, uint16 codes, radius 512
            std::string in_t  = (parts.size() > 1) ? parts[1] : "float";
            std::string cod_t = (parts.size() > 2) ? parts[2] : "uint16";
            const LorenzoParams& p = lorenzo_params_.count(sid)
                ? lorenzo_params_.at(sid) : default_lorenzo_;
            if(in_t == "float" && cod_t == "uint16") {
                auto* s = state_.ptr->addStage<fz::LorenzoStage<float, uint16_t>>();
                s->setErrorBound(error_bound_);
                s->setQuantRadius(p.quant_radius);
                s->setOutlierCapacity(p.outlier_capacity);
                return s;
            } else if(in_t == "double" && cod_t == "uint16") {
                auto* s = state_.ptr->addStage<fz::LorenzoStage<double, uint16_t>>();
                s->setErrorBound(error_bound_);
                s->setQuantRadius(p.quant_radius);
                s->setOutlierCapacity(p.outlier_capacity);
                return s;
            }
            throw std::runtime_error("Unsupported Lorenzo types: " + in_t + ":" + cod_t);
        }
        if(kind == "diff") {
            // Default: uint16 (most common after Lorenzo uint16 codes)
            std::string t = (parts.size() > 1) ? parts[1] : "uint16";
            if(t == "float")   return state_.ptr->addStage<fz::DifferenceStage<float>>();
            if(t == "double")  return state_.ptr->addStage<fz::DifferenceStage<double>>();
            if(t == "uint8")   return state_.ptr->addStage<fz::DifferenceStage<uint8_t>>();
            if(t == "uint16")  return state_.ptr->addStage<fz::DifferenceStage<uint16_t>>();
            if(t == "uint32")  return state_.ptr->addStage<fz::DifferenceStage<uint32_t>>();
            if(t == "int32")   return state_.ptr->addStage<fz::DifferenceStage<int32_t>>();
            if(t == "int64")   return state_.ptr->addStage<fz::DifferenceStage<int64_t>>();
            throw std::runtime_error("Unsupported diff type: " + t);
        }
        if(kind == "rle") {
            // Default: uint16
            std::string t = (parts.size() > 1) ? parts[1] : "uint16";
            if(t == "uint8")   return state_.ptr->addStage<fz::RLEStage<uint8_t>>();
            if(t == "uint16")  return state_.ptr->addStage<fz::RLEStage<uint16_t>>();
            if(t == "uint32")  return state_.ptr->addStage<fz::RLEStage<uint32_t>>();
            if(t == "int32")   return state_.ptr->addStage<fz::RLEStage<int32_t>>();
            throw std::runtime_error("Unsupported rle type: " + t);
        }
        if(kind == "passthrough") return state_.ptr->addStage<fz::PassThroughStage>();
        if(kind == "scale")       return state_.ptr->addStage<fz::ScaleStage>();
        throw std::runtime_error("Unknown stage type: " + token);
    }
    
    void build_pipeline(size_t data_size) {
        fz::MemoryStrategy strategy = fz::MemoryStrategy::PIPELINE;
        if(auto it = MEMORY_STRATEGIES.find(memory_strategy_); it != MEMORY_STRATEGIES.end())
            strategy = it->second;
        
        state_.ptr.reset(new fz::Pipeline(data_size, strategy, memory_multiplier_));
        
        // Create all stages and record by id (s0, s1, ...)
        std::map<std::string, fz::Stage*> ptrs;
        for(size_t i = 0; i < stages_.size(); i++) {
            std::string sid = std::format("s{}", i);
            ptrs[sid] = add_stage_from_token(stages_[i], sid);
        }
        
        // Wire connections: "s1 <- s0:codes"
        for(auto& conn_str : connections_) {
            auto c = parse_connection(conn_str);
            if(c.port.empty())
                state_.ptr->connect(ptrs.at(c.to), ptrs.at(c.from));
            else
                state_.ptr->connect(ptrs.at(c.to), ptrs.at(c.from), c.port);
        }
        
        state_.ptr->finalize();
    }

    // =========================================================================
    // Metadata Methods
    // =========================================================================
    
    int major_version() const override { 
        // TODO: update
        return 0; 
    }
    
    int minor_version() const override { 
        // TODO: update
        return 0; 
    }
    
    int patch_version() const override { 
        // TODO: update
        return 1; 
    }
    
    const char* version() const override { 
        // TODO: update
        return "0.0.1"; 
    }
    
    const char* prefix() const override { 
        return "fzgpumodules"; 
    }
    
    std::shared_ptr<libpressio_compressor_plugin> clone() override {
        return compat::make_unique<fzgpumodules_plugin>(*this);
    }
    
    pressio_options get_metrics_results_impl() const override {
        pressio_options metrics;
        set(metrics, "fzgpumodules:peak_memory", last_peak_memory_);
        set(metrics, "fzgpumodules:execution_time_us", last_execution_time_us_);
        return metrics;
    }

private:
    // =========================================================================
    // Lorenzo Options Helpers
    // =========================================================================

    void populate_lorenzo_options(pressio_options& options) const {
        for (size_t i = 0; i < stages_.size(); i++) {
            auto parts = split_str(stages_[i], ':');
            if (parts[0] != "lorenzo") continue;
            std::string sid = std::format("s{}", i);
            auto it = lorenzo_params_.find(sid);
            const LorenzoParams& p = (it != lorenzo_params_.end()) ? it->second : default_lorenzo_;
            set(options, std::format("fzgpumodules:{}:quant_radius", sid), p.quant_radius);
            set(options, std::format("fzgpumodules:{}:outlier_capacity", sid), p.outlier_capacity);
        }
    }

    void read_lorenzo_options(pressio_options const& options) {
        for (size_t i = 0; i < stages_.size(); i++) {
            auto parts = split_str(stages_[i], ':');
            if (parts[0] != "lorenzo") continue;
            std::string sid = std::format("s{}", i);
            if (!lorenzo_params_.count(sid))
                lorenzo_params_[sid] = LorenzoParams{};
            auto& p = lorenzo_params_[sid];
            get(options, std::format("fzgpumodules:{}:quant_radius", sid), &p.quant_radius);
            get(options, std::format("fzgpumodules:{}:outlier_capacity", sid), &p.outlier_capacity);
        }
    }

    void populate_lorenzo_documentation(pressio_options& options) const {
        for (size_t i = 0; i < stages_.size(); i++) {
            auto parts = split_str(stages_[i], ':');
            if (parts[0] != "lorenzo") continue;
            std::string sid = std::format("s{}", i);
            set(options, std::format("fzgpumodules:{}:quant_radius", sid),
                "Lorenzo quantization radius (controls bin count; larger = more bins, less error but more data)");
            set(options, std::format("fzgpumodules:{}:outlier_capacity", sid),
                "Lorenzo outlier reserve as fraction of total elements (e.g. 0.15 = 15% reserved for outliers)");
        }
    }

    // =========================================================================
    // Member Variables
    // =========================================================================

    // Pipeline state (CudaStream and PipelineState provide correct copy/move semantics)
    CudaStream stream_;
    PipelineState state_;
    
    // Plugin-level error bound
    double error_bound_ = 1e-3;

    // Global pipeline settings
    std::string memory_strategy_ = "minimal";
    float memory_multiplier_ = 3.0f;
    
    // Pipeline topology (user-configurable via options)
    std::vector<std::string> stages_      = {"lorenzo:float:uint16", "diff:uint16"};
    std::vector<std::string> connections_ = {"s1 <- s0:codes"};
    
    // TODO: per-stage settings -- when more modules will need more general approach
    // Per-stage Lorenzo parameters

    struct LorenzoParams {
        int   quant_radius      = 512;
        float outlier_capacity  = 0.15f;
        auto operator<=>(const LorenzoParams&) const = default;
    };
    const LorenzoParams default_lorenzo_{512, 0.15f};
    std::map<std::string, LorenzoParams> lorenzo_params_;
    
    // Metrics / state from last compression
    size_t last_peak_memory_ = 0;
    int64_t last_execution_time_us_ = 0;
};

// =============================================================================
// Plugin Registration
// =============================================================================

pressio_register registration(
    compressor_plugins(), 
    "fzgpumodules", 
    []() { 
        return compat::make_unique<fzgpumodules_plugin>(); 
    }
);

} } } // namespace libpressio::compressors::fzgpumodules_ns
