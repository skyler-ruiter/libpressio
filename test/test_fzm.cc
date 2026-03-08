// test_fzm.cc  –  FZGpuModules plugin tests
//
// Covers two pipeline topologies:
//   BasicLinearPipeline  : lorenzo:float:uint16  ->  diff:uint16
//   ParallelDAGPipeline  : lorenzo:float:uint16  ->  diff:uint16  (codes)
//                                                ->  passthrough  (outlier_errors)
//
// Control via environment variables:
//   LIBPRESSIO_TEST_VERBOSE   – if set, print options/metrics to stdout

#include <cmath>
#include <gtest/gtest.h>

#include "libpressio.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"

// ── helpers ──────────────────────────────────────────────────────────────────

static bool verbose() { return std::getenv("LIBPRESSIO_TEST_VERBOSE") != nullptr; }

// Generate a smooth 2-D float field of size rows x cols
static pressio_data make_smooth_data(size_t rows = 256, size_t cols = 256) {
    pressio_data d = pressio_data::owning(pressio_float_dtype, {cols, rows});
    float* p = static_cast<float*>(d.data());
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            *p++ = std::sin(2.0f * float(M_PI) * r / rows)
                 + std::cos(4.0f * float(M_PI) * c / cols)
                 + 0.5f * std::sin(float(M_PI) * (r + c) / (rows + cols));
    return d;
}

// Build a composite metrics bundle 
static pressio_metrics make_metrics(pressio& lib) {
    std::vector<std::string> names = {"error_stat", "size"};
    pressio probe;
    for (const char* id : {"time", "memory"}) {
        if (probe.get_metric(id)) names.push_back(id);
    }
    return lib.get_metrics(names.begin(), names.end());
}

struct RunResult {
    double   compression_ratio = 0; // size:compression_ratio
    double   max_error         = 0; // error_stat:max_error
    double   psnr              = 0; // error_stat:psnr
    double   mse               = 0; // error_stat:mse
    double   compress_time_ms  = 0; // time:compress        (0 if module absent)
    double   throughput_mb_s   = 0; // derived from compress_time_ms
    uint64_t memory_compress   = 0; // memory:compress      (0 if module absent)
};

// compress+decompress round-trip
static RunResult run_pipeline(pressio&              lib,
                              const pressio_options& opts,
                              pressio_data&          input) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    if (!comp) {
        ADD_FAILURE() << "could not load fzgpumodules: " << lib.err_msg();
        return {};
    }

    auto m = make_metrics(lib);
    if (!m) {
        ADD_FAILURE() << "could not build composite metrics: " << lib.err_msg();
        return {};
    }
    comp->set_metrics(m);

    if (comp->set_options(opts)) {
        ADD_FAILURE() << "set_options failed: " << comp->error_msg();
        return {};
    }

    if (verbose()) {
        auto cur = comp->get_options();
        std::vector<std::string> stages, conns;
        std::string strat; float mult = 0;
        cur.get("fzgpumodules:stages",           &stages);
        cur.get("fzgpumodules:connections",       &conns);
        cur.get("fzgpumodules:memory_strategy",   &strat);
        cur.get("fzgpumodules:memory_multiplier", &mult);
        std::cout << "  stages:   ";
        for (auto& s : stages) std::cout << s << "  ";
        std::cout << "\n  conns:    ";
        for (auto& c : conns)  std::cout << c << "  ";
        std::cout << "\n  memory:   " << strat << " x" << mult << "\n";
    }

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    if (comp->compress(&input, &compressed)) {
        ADD_FAILURE() << "compress failed: " << comp->error_msg();
        return {};
    }

    pressio_data output = pressio_data::owning(pressio_float_dtype, input.dimensions());
    if (comp->decompress(&compressed, &output)) {
        ADD_FAILURE() << "decompress failed: " << comp->error_msg();
        return {};
    }

    auto results = comp->get_metrics_results();
    if (verbose()) {
        double cr = 0, max_err = 0, psnr = 0;
        uint64_t comp_sz = 0, uncomp_sz = 0;
        results.get("size:compression_ratio", &cr);
        results.get("size:compressed_size",   &comp_sz);
        results.get("size:uncompressed_size", &uncomp_sz);
        results.get("error_stat:max_error",   &max_err);
        results.get("error_stat:psnr",        &psnr);
        std::cout << "  ratio:    " << cr
                  << "  (" << uncomp_sz << " -> " << comp_sz << " bytes)\n"
                  << "  max_err:  " << max_err << "\n"
                  << "  psnr:     " << psnr << " dB\n";
    }

    RunResult r;
    double   cr = 0, max_err = 0, psnr = 0, mse = 0, t_ms = 0;
    uint64_t mem = 0;
    results.get("size:compression_ratio", &cr);
    results.get("error_stat:max_error",   &max_err);
    results.get("error_stat:psnr",        &psnr);
    results.get("error_stat:mse",         &mse);
    results.get("time:compress",          &t_ms);
    results.get("memory:compress",        &mem);

    r.compression_ratio = cr;
    r.max_error         = max_err;
    r.psnr              = psnr;
    r.mse               = mse;
    r.compress_time_ms  = t_ms;
    r.memory_compress   = mem;
    if (t_ms > 0)
        r.throughput_mb_s =
            (input.size_in_bytes() / (1024.0 * 1024.0)) / (t_ms / 1000.0);
    return r;
}

// ── test fixture ──────────────────────────────────────────────────────────────

class FzgpuModulesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!lib.get_compressor("fzgpumodules"))
            GTEST_SKIP() << "fzgpumodules not available in this build";

        input = make_smooth_data();
    }

    pressio      lib;
    pressio_data input;
};

// ── TEST: linear pipeline ─────────────────────────────────────────────────────
//
//  s0: lorenzo:float:uint16
//  s1: diff:uint16  <-  s0:codes

TEST_F(FzgpuModulesTest, BasicLinearPipeline) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs",                       eb);
    opts.set("fzgpumodules:stages",
             std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections",
             std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:s0:quant_radius",      512);
    opts.set("fzgpumodules:s0:outlier_capacity",  0.15f);
    opts.set("fzgpumodules:memory_strategy",      std::string("minimal"));
    opts.set("fzgpumodules:memory_multiplier",    3.0f);

    auto r = run_pipeline(lib, opts, input);

    EXPECT_GT(r.compression_ratio, 1.0);
    EXPECT_LE(r.max_error, eb + 1e-5);
}

// ── TEST: parallel-branch DAG ─────────────────────────────────────────────────
//
//  s0: lorenzo:float:uint16
//  s1: diff:uint16   <-  s0:codes          (difference coding of quant codes)
//  s2: passthrough   <-  s0:outlier_errors (identity path for outlier errors)

TEST_F(FzgpuModulesTest, ParallelDAGPipeline) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs",                       eb);
    opts.set("fzgpumodules:stages",
             std::vector<std::string>{
                 "lorenzo:float:uint16", "diff:uint16", "passthrough"});
    opts.set("fzgpumodules:connections",
             std::vector<std::string>{
                 "s1 <- s0:codes",
                 "s2 <- s0:outlier_errors"});
    opts.set("fzgpumodules:s0:quant_radius",      32);
    opts.set("fzgpumodules:s0:outlier_capacity",  0.05f);
    opts.set("fzgpumodules:memory_strategy",      std::string("pipeline"));
    opts.set("fzgpumodules:memory_multiplier",    3.0f);

    auto r = run_pipeline(lib, opts, input);

    EXPECT_GT(r.compression_ratio, 1.0);
    EXPECT_LE(r.max_error, eb + 1e-5);
}
