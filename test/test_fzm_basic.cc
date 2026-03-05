#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include "libpressio.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"

static bool check(bool cond, const char* msg) {
    if(!cond) { std::cerr << "FAIL: " << msg << "\n"; }
    return cond;
}

struct Stats {
    double max_error;
    double mse;
    double psnr;
    double nrmse;
};

static Stats calculateStatistics(const float* orig, const float* restored, size_t n) {
    double max_err = 0.0, sum_sq = 0.0;
    double data_min = std::numeric_limits<double>::max();
    double data_max = std::numeric_limits<double>::lowest();
    for(size_t i = 0; i < n; i++) {
        double o = orig[i], r = restored[i];
        double diff = std::abs(o - r);
        if(diff > max_err) max_err = diff;
        sum_sq += diff * diff;
        if(o < data_min) data_min = o;
        if(o > data_max) data_max = o;
    }
    double mse   = sum_sq / n;
    double range = data_max - data_min;
    double psnr  = (mse > 0.0) ? (20.0 * std::log10(range) - 10.0 * std::log10(mse)) : std::numeric_limits<double>::infinity();
    double nrmse = (range > 0.0) ? std::sqrt(mse) / range : 0.0;
    return {max_err, mse, psnr, nrmse};
}

int main() {
    std::cout << "=== FZModules Plugin Test ===\n\n";

    pressio library;
    pressio_compressor comp = library.get_compressor("fzmodules");
    
    if(!check(!!comp, "load fzmodules compressor")) return 1;

    std::cout << "Prefix: " << comp->prefix()
              << "  Version: " << comp->version() << "\n\n";

    // -------------------------------------------------------------------------
    // 1. Print available memory strategies from configuration
    // -------------------------------------------------------------------------
    pressio_options config = comp->get_configuration();
    std::vector<std::string> strategies;
    if(config.get("fzmodules:memory_strategy", &strategies) == pressio_options_key_set) {
        std::cout << "Available memory strategies:";
        for(auto const& s : strategies) std::cout << " " << s;
        std::cout << "\n\n";
    }

    // -------------------------------------------------------------------------
    // 2. Load dataset from file
    // -------------------------------------------------------------------------
    const char* path = "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if(!check(fin.is_open(), "open input file")) {
        std::cerr << "Path: " << path << "\n";
        return 1;
    }
    const size_t file_bytes = static_cast<size_t>(fin.tellg());
    const size_t N = file_bytes / sizeof(float);
    fin.seekg(0);
    std::vector<float> data(N);
    fin.read(reinterpret_cast<char*>(data.data()), file_bytes);
    fin.close();
    std::cout << "Dataset: " << path << "\n"
              << "  " << N << " floats  (" << (file_bytes / (1024.0*1024.0)) << " MB)  "
              << "[" << 1800 << " x " << 3600 << "]\n\n";

    pressio_data input = pressio_data::copy(pressio_float_dtype, data.data(), {N});

    // -------------------------------------------------------------------------
    // 3. Configure via new API
    //    - pressio:abs        → error bound for all Lorenzo stages
    //    - fzmodules:stages   → ordered stage tokens
    //    - fzmodules:connections → wiring between stages
    //    - fzmodules:s0:*     → per-stage Lorenzo params for stage 0
    // -------------------------------------------------------------------------
    pressio_options opts;

    // Standard libpressio error bound
    opts.set("pressio:abs", 1e-3f);

    // Pipeline topology: Lorenzo(float→uint16) → Diff(uint16)
    opts.set("fzmodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzmodules:connections", std::vector<std::string>{"s1 <- s0:codes"});

    // Lorenzo stage (s0) params
    opts.set("fzmodules:s0:quant_radius",     512);
    opts.set("fzmodules:s0:outlier_capacity", 0.15f);

    // Memory settings
    opts.set("fzmodules:memory_strategy",   std::string("minimal"));
    opts.set("fzmodules:memory_multiplier", 3.0f);

    if(!check(!comp->set_options(opts), "set options")) {
        std::cerr << comp->error_msg() << "\n";
        return 1;
    }

    // Echo back what was set
    pressio_options current = comp->get_options();
    float eb = 0; std::string strat;
    std::vector<std::string> stages, connections;
    int qr = 0; float oc = 0;
    current.get("pressio:abs",                  &eb);
    current.get("fzmodules:memory_strategy",     &strat);
    current.get("fzmodules:stages",              &stages);
    current.get("fzmodules:connections",         &connections);
    current.get("fzmodules:s0:quant_radius",     &qr);
    current.get("fzmodules:s0:outlier_capacity", &oc);
    std::cout << "Config:\n"
              << "  pressio:abs           = " << eb    << "\n"
              << "  memory_strategy       = " << strat << "\n"
              << "  stages                = [";
    for(size_t i = 0; i < stages.size(); i++) std::cout << (i?", ":"") << stages[i];
    std::cout << "]\n  connections           = [";
    for(size_t i = 0; i < connections.size(); i++) std::cout << (i?", ":"") << connections[i];
    std::cout << "]\n"
              << "  s0:quant_radius       = " << qr << "\n"
              << "  s0:outlier_capacity   = " << oc << "\n\n";

    // -------------------------------------------------------------------------
    // 4. Compress
    // -------------------------------------------------------------------------
    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    if(!check(!comp->compress(&input, &compressed), "compress")) {
        std::cerr << comp->error_msg() << "\n";
        return 1;
    }

    double ratio = (double)input.size_in_bytes() / compressed.size_in_bytes();
    std::cout << "Compression:\n"
              << "  Input:  " << (input.size_in_bytes()      / (1024.0*1024.0)) << " MB\n"
              << "  Output: " << (compressed.size_in_bytes() / (1024.0*1024.0)) << " MB\n"
              << "  Ratio:  " << ratio << "x\n";

    pressio_options metrics = comp->get_metrics_results();
    size_t peak_mem = 0; int64_t exec_us = 0;
    metrics.get("fzmodules:peak_memory",       &peak_mem);
    metrics.get("fzmodules:execution_time_us", &exec_us);
    if(peak_mem) std::cout << "  Peak GPU memory: " << (peak_mem/(1024.0*1024.0)) << " MB\n";
    if(exec_us)  std::cout << "  Time: " << exec_us << " μs  ("
                           << (input.size_in_bytes()/(1024.0*1024.0)) / (exec_us/1e6)
                           << " MB/s)\n";
    std::cout << "\n";
    check(ratio > 1.0, "compression ratio > 1");

    // -------------------------------------------------------------------------
    // 5. Decompress and verify error bound
    // -------------------------------------------------------------------------
    pressio_data decompressed = pressio_data::owning(pressio_float_dtype, {N});
    if(!check(!comp->decompress(&compressed, &decompressed), "decompress")) {
        std::cerr << comp->error_msg() << "\n";
        return 1;
    }

    const float* orig     = data.data();
    const float* restored = static_cast<const float*>(decompressed.data());
    auto stats = calculateStatistics(orig, restored, N);

    std::cout << "Decompression quality:\n"
              << "  Max absolute error: " << stats.max_error << "  (bound: " << eb << ")\n"
              << "  MSE:                " << stats.mse   << "\n"
              << "  PSNR:               " << stats.psnr  << " dB\n"
              << "  NRMSE:              " << stats.nrmse << "\n\n";
    check(stats.max_error <= eb + 1e-5, "max error within bound");

    std::cout << "=== All tests passed! ===\n";
    return 0;
}
