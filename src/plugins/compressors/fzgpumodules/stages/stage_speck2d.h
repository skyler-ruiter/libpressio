#pragma once
#include "stage_kind.h"
#include "coders/speck2d/speck2d_stage.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// GPU-parallel "wavefront" SPECK-like bit-plane coder (2-D only). Takes
// signed int32 codes (the same convention QuantizerStage's linear_mode
// emits), no per-stage options. Not a port of SPERR's SPECK bitstream --
// see FZGPUModules' THIRD_PARTY.md and docs/stages/speck2d.md.
class Speck2DStageKind : public CloneableStageKind<Speck2DStageKind> {
public:
    bool matches(std::string_view kind) const override { return kind == "speck2d"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        return ctx.pipeline.addStage<fz::Speck2DStage>();
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
