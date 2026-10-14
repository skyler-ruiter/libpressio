#pragma once
#include "stage_kind.h"
#include "coders/cdf97_outlier_correct/cdf97_outlier_correct_stage.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// Sparse outlier correction that turns a coefficient-domain quantization
// bound into an actually GUARANTEED reconstructed-domain pointwise error
// bound (see FZGPUModules' modules/coders/outlier_correct/
// outlier_correct_stage.h for the mechanism). error_bound is read from the
// SAME ctx.eb every other lossy stage uses, so it automatically matches
// whatever paired QuantizerStage token appears in the same pipeline -- no
// separate parameter to keep in sync by hand.
//
// This stage needs the pipeline's raw input bound directly to it (see
// Pipeline::bindExternalInput() in FZGPUModules) -- that's a WIRING concern,
// done via a "sN <- __external__" connection string in
// fzgpumodules:connections, not here. See docs/libpressio_python.md's SPERR
// recipe for the full pipeline.
class Cdf97OutlierCorrectStageKind : public CloneableStageKind<Cdf97OutlierCorrectStageKind> {
public:
    bool matches(std::string_view kind) const override { return kind == "cdf97_outlier_correct"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        auto* s = ctx.pipeline.addStage<fz::Cdf97OutlierCorrectStage>();
        s->setErrorBound(static_cast<float>(ctx.eb));
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
