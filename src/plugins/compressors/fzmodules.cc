#include <vector>
#include <memory>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include "std_compat/memory.h"
#include "std_compat/utility.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

#include "fzmodules.h"
#include <cuda_runtime.h>

namespace libpressio { namespace compressors { namespace fzmodules_ns {

// =============================================================================
// Helper Functions & Constants
// =============================================================================

// Memory strategy mapping for FZModules Pipeline
static const std::map<std::string, fz::MemoryStrategy> MEMORY_STRATEGIES {
    {"preallocate", fz::MemoryStrategy::PREALLOCATE}, // speed at cost of memory usage
    {"pipeline", fz::MemoryStrategy::PIPELINE}, // critical path preallocate and free when done
    {"minimal", fz::MemoryStrategy::MINIMAL}, // only allocate what is needed for each stage at execution time
};

// Helper to extract keys from maps for configuration
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

// Split "a:b:c" -> {"a","b","c"}
static std::vector<std::string> split_str(const std::string& s, char delim) {
    std::vector<std::string> parts;
    size_t start = 0, pos;
    while((pos = s.find(delim, start)) != std::string::npos) {
        parts.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    parts.push_back(s.substr(start));
    return parts;
}

static std::string trim_str(const std::string& s) {
    size_t b = s.find_first_not_of(" \t");
    if(b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

// Parsed connection from "sN <- sM:port" or "sN <- sM"
struct ParsedConnection {
    std::string to, from, port;
};

static ParsedConnection parse_connection(const std::string& s) {
    auto arrow = s.find(" <- ");
    if(arrow == std::string::npos)
        throw std::invalid_argument("Invalid connection (expected 'sN <- sM[:port]'): " + s);
    ParsedConnection c;
    c.to = trim_str(s.substr(0, arrow));
    std::string rhs = trim_str(s.substr(arrow + 4));
    auto colon = rhs.find(':');
    if(colon == std::string::npos) { c.from = rhs; }
    else { c.from = rhs.substr(0, colon); c.port = rhs.substr(colon + 1); }
    return c;
}

// =============================================================================
// Main Plugin Class
// =============================================================================

class fzmodules_plugin : public libpressio_compressor_plugin {
public:
    // =========================================================================
    // Constructor & Destructor
    // =========================================================================
    
    fzmodules_plugin() : pipeline_(nullptr), pipeline_dirty_(true), error_bound_(1e-3f) {
        // Seed default Lorenzo params for s0
        lorenzo_params_["s0"] = LorenzoParams{};
        cudaStreamCreate(&stream_);
    }
    
    ~fzmodules_plugin() {
        if(stream_) cudaStreamDestroy(stream_);
    }
    
    // Copy constructor
    fzmodules_plugin(fzmodules_plugin const& rhs)
        : libpressio_compressor_plugin(rhs),
          pipeline_(nullptr),  // Force rebuild
          pipeline_dirty_(true)
    { cudaStreamCreate(&stream_); copy_config_from(rhs); }

    // Move constructor
    fzmodules_plugin(fzmodules_plugin&& rhs) noexcept
        : libpressio_compressor_plugin(std::move(rhs)),
          pipeline_(std::move(rhs.pipeline_)),
          pipeline_dirty_(rhs.pipeline_dirty_),
          stream_(rhs.stream_)
    {
        rhs.stream_ = nullptr;
        move_config_from(rhs);
        rhs.pipeline_dirty_ = true;
    }

    // Assignment operators
    fzmodules_plugin& operator=(fzmodules_plugin const& rhs) {
        if(this == &rhs) return *this;
        libpressio_compressor_plugin::operator=(rhs);
        pipeline_.reset();
        pipeline_dirty_ = true;
        if(!stream_) cudaStreamCreate(&stream_);
        copy_config_from(rhs);
        return *this;
    }

    fzmodules_plugin& operator=(fzmodules_plugin&& rhs) noexcept {
        if(this == &rhs) return *this;
        libpressio_compressor_plugin::operator=(std::move(rhs));
        pipeline_ = std::move(rhs.pipeline_);
        pipeline_dirty_ = rhs.pipeline_dirty_;
        if(stream_) cudaStreamDestroy(stream_);
        stream_ = rhs.stream_;
        rhs.stream_ = nullptr;
        move_config_from(rhs);
        rhs.pipeline_dirty_ = true;
        return *this;
    }

    // =========================================================================
    // Options Interface
    // =========================================================================
    
    struct pressio_options get_options_impl() const override {
        struct pressio_options options;
        
        // TODO: relative error bound mode
        // Standard libpressio error bound
        set(options, "pressio:abs", error_bound_);
        
        // Global pipeline settings
        set(options, "fzmodules:memory_strategy", memory_strategy_);
        set(options, "fzmodules:memory_multiplier", memory_multiplier_);
        
        // Pipeline topology
        set(options, "fzmodules:stages", stages_);
        set(options, "fzmodules:connections", connections_);
        
        // TODO: per-stage settings -- when more modules will need more general approach
        // Per-stage Lorenzo parameters (quant_radius and outlier_capacity)
        for(size_t i = 0; i < stages_.size(); i++) {
            auto parts = split_str(stages_[i], ':');
            if(parts[0] != "lorenzo") continue;
            std::string sid = "s" + std::to_string(i);
            auto it = lorenzo_params_.find(sid);
            const LorenzoParams& p = (it != lorenzo_params_.end()) ? it->second : default_lorenzo_;
            set(options, "fzmodules:" + sid + ":quant_radius", p.quant_radius);
            set(options, "fzmodules:" + sid + ":outlier_capacity", p.outlier_capacity);
        }
        
        // Metrics from last run
        if(last_peak_memory_ > 0)
            set(options, "fzmodules:peak_memory", last_peak_memory_);
        if(last_execution_time_us_ > 0)
            set(options, "fzmodules:execution_time_us", last_execution_time_us_);
        
        return options;
    }
    
    struct pressio_options get_configuration_impl() const override {
        struct pressio_options options;
        
        // Thread safety and stability
        set(options, "pressio:thread_safe", pressio_thread_safety_single);
        set(options, "pressio:stability", "experimental");
        
        // Tell libpressio that pressio:abs is a supported error-bound mode
        set(options, "pressio:abs", "lossy");
        
        // Available choices for string options
        set(options, "fzmodules:memory_strategy", map_keys(MEMORY_STRATEGIES));
        
        return options;
    }
    
    struct pressio_options get_documentation_impl() const override {
        struct pressio_options options;
        
        set(options, "pressio:description",
            R"(FZModules GPU compressor with user-composable pipeline.

Stage tokens (fzmodules:stages):
  lorenzo:float:uint16   Lorenzo predictor (float input, uint16 codes)
  lorenzo:double:uint16  Lorenzo predictor (double input, uint16 codes)
  diff:uint16            Difference coding over uint16 values
  diff:uint32            Difference coding over uint32 values
  diff:float             Difference coding over float values
  rle:uint16             Run-length encoding over uint16 values
  rle:uint32             Run-length encoding over uint32 values
  passthrough            Identity (copy) stage
  scale                  Scale stage

Connections (fzmodules:connections) - 'sN <- sM' or 'sN <- sM:port':
  "s1 <- s0:codes"       stage s1 reads the 'codes' output of stage s0
  "s1 <- s0"             stage s1 reads the default output of s0

Per-stage Lorenzo params use 'fzmodules:sN:key' (e.g. fzmodules:s0:quant_radius).)");
        
        set(options, "pressio:abs",
            "Absolute error bound applied to all Lorenzo stages in the pipeline");
        set(options, "fzmodules:memory_strategy",
            "Memory allocation strategy: 'preallocate', 'pipeline', or 'minimal'");
        set(options, "fzmodules:memory_multiplier",
            "Memory allocation multiplier (e.g., 3.0 = 3x input size)");
        set(options, "fzmodules:stages",
            "Ordered list of stage tokens defining the pipeline stages");
        set(options, "fzmodules:connections",
            "List of connections between stages: 'sN <- sM[:port]'");
        set(options, "fzmodules:sN:quant_radius",
            "Lorenzo quantization radius for stage sN (e.g. s0). Controls quantization bin count.");
        set(options, "fzmodules:sN:outlier_capacity",
            "Lorenzo outlier reserve factor for stage sN (e.g. s0). Fraction of elements reserved for outliers.");
        set(options, "fzmodules:peak_memory",
            "[Output] Peak GPU memory usage from last compression (bytes)");
        set(options, "fzmodules:execution_time_us",
            "[Output] Execution time from last compression (microseconds)");
        
        return options;
    }
    
    int set_options_impl(struct pressio_options const& options) override {
        // Standard error bound
        float old_error_bound = error_bound_;
        get(options, "pressio:abs", &error_bound_);
        
        // Topology
        auto old_stages = stages_;
        auto old_connections = connections_;
        get(options, "fzmodules:stages", &stages_);
        get(options, "fzmodules:connections", &connections_);
        
        // Global settings
        std::string old_strategy = memory_strategy_;
        float old_multiplier = memory_multiplier_;
        get(options, "fzmodules:memory_strategy", &memory_strategy_);
        get(options, "fzmodules:memory_multiplier", &memory_multiplier_);
        
        // TODO: per-stage settings -- when more modules will need more general approach
        // Per-stage Lorenzo params
        auto old_params = lorenzo_params_;
        for(size_t i = 0; i < stages_.size(); i++) {
            auto parts = split_str(stages_[i], ':');
            if(parts[0] != "lorenzo") continue;
            std::string sid = "s" + std::to_string(i);
            if(!lorenzo_params_.count(sid))
                lorenzo_params_[sid] = LorenzoParams{};
            auto& p = lorenzo_params_[sid];
            get(options, "fzmodules:" + sid + ":quant_radius", &p.quant_radius);
            get(options, "fzmodules:" + sid + ":outlier_capacity", &p.outlier_capacity);
        }
        
        if(stages_ != old_stages || connections_ != old_connections ||
           memory_strategy_ != old_strategy ||
           memory_multiplier_ != old_multiplier ||
           error_bound_ != old_error_bound ||
           lorenzo_params_ != old_params) {
            pipeline_dirty_ = true;
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
                    return set_error(1, "Unsupported data type for FZModules compression");
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
                    return set_error(1, "Unsupported data type for FZModules decompression");
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
        // Get input parameters
        const T* h_input = static_cast<const T*>(real_input->data());
        size_t data_size = real_input->size_in_bytes();
        
        // Allocate GPU memory and copy input data
        T* d_input = nullptr;
        cudaError_t err = cudaMalloc(&d_input, data_size);
        if(err != cudaSuccess) {
            return set_error(10, std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
        
        err = cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
            cudaFree(d_input);
            return set_error(11, std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err));
        }
        
        // Build pipeline if needed (also rebuild if input size has changed)
        if(pipeline_dirty_ || !pipeline_ || data_size != last_built_size_) {
            try {
                build_pipeline(data_size);
                pipeline_dirty_ = false;
                last_built_size_ = data_size;
            } catch (std::exception const& ex) {
                cudaFree(d_input);
                return set_error(3, std::string("Pipeline construction failed: ") + ex.what());
            }
        }
        
        // run compression with timing
        void* d_output = nullptr;
        size_t output_size = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            pipeline_->compress(d_input, data_size, &d_output, &output_size, stream_);
        } catch (std::exception const& ex) {
            cudaFree(d_input);
            return set_error(4, std::string("Compression execution failed: ") + ex.what());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        last_execution_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        last_peak_memory_ = pipeline_->getPeakMemoryUsage();
        
        // Free input GPU memory
        cudaFree(d_input);
        
        // Copy output back to host
        std::vector<uint8_t> host_output(output_size);
        err = cudaMemcpy(host_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
            cudaFree(d_output);
            return set_error(12, std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err));
        }
        
        // Free output GPU memory
        cudaFree(d_output);
        
        // Create output pressio_data with host memory
        *output = pressio_data::copy(
            pressio_byte_dtype,
            host_output.data(),
            {output_size}
        );
        
        return 0;
    }
    
    template<typename T>
    int decompress_typed(const pressio_data* compressed_input, struct pressio_data* output) {
        if(!pipeline_) {
            return set_error(5, "Decompression requires a pipeline built by a prior compress call");
        }
        const size_t original_size = output->size_in_bytes();
        const void* h_compressed = compressed_input->data();
        size_t compressed_size = compressed_input->size_in_bytes();
        
        // Copy compressed data to GPU
        void* d_compressed = nullptr;
        cudaError_t err = cudaMalloc(&d_compressed, compressed_size);
        if(err != cudaSuccess) {
            return set_error(10, std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
        err = cudaMemcpy(d_compressed, h_compressed, compressed_size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
            cudaFree(d_compressed);
            return set_error(11, std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err));
        }
        
        void* d_output = nullptr;
        size_t output_size = 0;
        
        try {
            // decompress() runs on same pipeline as compress
            // second arg is the ORIGINAL data size, not compressed_size.
            pipeline_->decompress(d_compressed, original_size, &d_output, &output_size, stream_);
        } catch(std::exception const& ex) {
            cudaFree(d_compressed);
            return set_error(4, std::string("Decompression execution failed: ") + ex.what());
        }
        
        cudaFree(d_compressed);
        
        // Copy result back to host
        std::vector<uint8_t> host_output(output_size);
        err = cudaMemcpy(host_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);
        cudaFree(d_output);
        if(err != cudaSuccess) {
            return set_error(12, std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err));
        }
        
        *output = pressio_data::copy(
            output->dtype(),
            host_output.data(),
            output->dimensions()
        );
        
        return 0;
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    // Create a stage from a token (e.g. "lorenzo:float:uint16"), add to pipeline, return ptr
    // essentially a factory for stages based on string tokens
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
                auto* s = pipeline_->addStage<fz::LorenzoStage<float, uint16_t>>();
                s->setErrorBound(error_bound_);
                s->setQuantRadius(p.quant_radius);
                s->setOutlierCapacity(p.outlier_capacity);
                return s;
            } else if(in_t == "double" && cod_t == "uint16") {
                auto* s = pipeline_->addStage<fz::LorenzoStage<double, uint16_t>>();
                s->setErrorBound(static_cast<double>(error_bound_));
                s->setQuantRadius(p.quant_radius);
                s->setOutlierCapacity(p.outlier_capacity);
                return s;
            }
            throw std::runtime_error("Unsupported Lorenzo types: " + in_t + ":" + cod_t);
        }
        if(kind == "diff") {
            // Default: uint16 (most common after Lorenzo uint16 codes)
            std::string t = (parts.size() > 1) ? parts[1] : "uint16";
            if(t == "float")   return pipeline_->addStage<fz::DifferenceStage<float>>();
            if(t == "double")  return pipeline_->addStage<fz::DifferenceStage<double>>();
            if(t == "uint8")   return pipeline_->addStage<fz::DifferenceStage<uint8_t>>();
            if(t == "uint16")  return pipeline_->addStage<fz::DifferenceStage<uint16_t>>();
            if(t == "uint32")  return pipeline_->addStage<fz::DifferenceStage<uint32_t>>();
            if(t == "int32")   return pipeline_->addStage<fz::DifferenceStage<int32_t>>();
            if(t == "int64")   return pipeline_->addStage<fz::DifferenceStage<int64_t>>();
            throw std::runtime_error("Unsupported diff type: " + t);
        }
        if(kind == "rle") {
            // Default: uint16
            std::string t = (parts.size() > 1) ? parts[1] : "uint16";
            if(t == "uint8")   return pipeline_->addStage<fz::RLEStage<uint8_t>>();
            if(t == "uint16")  return pipeline_->addStage<fz::RLEStage<uint16_t>>();
            if(t == "uint32")  return pipeline_->addStage<fz::RLEStage<uint32_t>>();
            if(t == "int32")   return pipeline_->addStage<fz::RLEStage<int32_t>>();
            throw std::runtime_error("Unsupported rle type: " + t);
        }
        if(kind == "passthrough") return pipeline_->addStage<fz::PassThroughStage>();
        if(kind == "scale")       return pipeline_->addStage<fz::ScaleStage>();
        throw std::runtime_error("Unknown stage type: " + token);
    }
    
    void build_pipeline(size_t data_size) {
        fz::MemoryStrategy strategy = fz::MemoryStrategy::PIPELINE;
        if(auto it = MEMORY_STRATEGIES.find(memory_strategy_); it != MEMORY_STRATEGIES.end())
            strategy = it->second;
        
        pipeline_.reset(new fz::Pipeline(data_size, strategy, memory_multiplier_));
        
        // Create all stages and record by id (s0, s1, ...)
        std::map<std::string, fz::Stage*> ptrs;
        for(size_t i = 0; i < stages_.size(); i++) {
            std::string sid = "s" + std::to_string(i);
            ptrs[sid] = add_stage_from_token(stages_[i], sid);
        }
        
        // Wire connections: "s1 <- s0:codes"
        for(auto& conn_str : connections_) {
            auto c = parse_connection(conn_str);
            if(c.port.empty())
                pipeline_->connect(ptrs.at(c.to), ptrs.at(c.from));
            else
                pipeline_->connect(ptrs.at(c.to), ptrs.at(c.from), c.port);
        }
        
        pipeline_->finalize();
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
        return "fzmodules"; 
    }
    
    std::shared_ptr<libpressio_compressor_plugin> clone() override {
        return compat::make_unique<fzmodules_plugin>(*this);
    }
    
    pressio_options get_metrics_results_impl() const override {
        pressio_options metrics;
        if(last_peak_memory_ > 0) {
            set(metrics, "fzmodules:peak_memory", last_peak_memory_);
        }
        if(last_execution_time_us_ > 0) {
            set(metrics, "fzmodules:execution_time_us", last_execution_time_us_);
        }
        return metrics;
    }

private:
    // =========================================================================
    // Config Copy/Move Helpers
    // =========================================================================

    void copy_config_from(fzmodules_plugin const& rhs) {
        error_bound_       = rhs.error_bound_;
        memory_strategy_   = rhs.memory_strategy_;
        memory_multiplier_ = rhs.memory_multiplier_;
        stages_            = rhs.stages_;
        connections_       = rhs.connections_;
        lorenzo_params_    = rhs.lorenzo_params_;
    }

    void move_config_from(fzmodules_plugin& rhs) {
        error_bound_       = rhs.error_bound_;
        memory_strategy_   = std::move(rhs.memory_strategy_);
        memory_multiplier_ = rhs.memory_multiplier_;
        stages_            = std::move(rhs.stages_);
        connections_       = std::move(rhs.connections_);
        lorenzo_params_    = std::move(rhs.lorenzo_params_);
    }

    // =========================================================================
    // Member Variables
    // =========================================================================

    // Pipeline state
    std::unique_ptr<fz::Pipeline> pipeline_;
    bool pipeline_dirty_;
    cudaStream_t stream_ = nullptr;
    
    // Plugin-level error bound
    float error_bound_ = 1e-3f;

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
        bool operator==(const LorenzoParams& o) const {
            return quant_radius     == o.quant_radius &&
                   outlier_capacity == o.outlier_capacity;
        }
        bool operator!=(const LorenzoParams& o) const { return !(*this == o); }
    };
    inline static constexpr LorenzoParams default_lorenzo_ = {512, 0.15f};
    std::map<std::string, LorenzoParams> lorenzo_params_;
    
    // Metrics / state from last compression
    size_t last_built_size_ = 0;       // data_size used when pipeline was last built
    size_t last_peak_memory_ = 0;
    int64_t last_execution_time_us_ = 0;
};

// =============================================================================
// Plugin Registration
// =============================================================================

pressio_register registration(
    compressor_plugins(), 
    "fzmodules", 
    []() { 
        return compat::make_unique<fzmodules_plugin>(); 
    }
);

} } } // namespace libpressio::compressors::fzmodules_ns
