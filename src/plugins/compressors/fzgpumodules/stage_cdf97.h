#pragma once
#include "stage_kind.h"
#include "transforms/cdf97/cdf97_stage.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// CDF 9/7 biorthogonal wavelet transform (SPERR's DWT front-half). Lossless,
// invertible, size-preserving float -> float basis change -- feeds a
// Quantizer, not lossy itself, so unlike quantizer/lorenzo this stage does
// NOT read ctx.eb. See docs/stages/cdf97.md in FZGPUModules for the current
// long-line (shared-memory) size limitation.
class Cdf97StageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "cdf97"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string t = (parts.size() > 1) ? parts[1] : "float";

        if(t == "float")  return ctx.pipeline.addStage<fz::Cdf97Stage<float>>();
        if(t == "double") return ctx.pipeline.addStage<fz::Cdf97Stage<double>>();
        throw std::runtime_error("Unsupported cdf97 type: " + t);
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
