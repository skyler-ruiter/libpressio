#pragma once
#include <compare>
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct BitshuffleParams {
    int element_width = 4;      // bytes: 1, 2, 4, or 8; must match incoming data element type
    int block_size    = 16384;  // bytes; must be multiple of 1024*element_width
    auto operator<=>(const BitshuffleParams&) const = default;
};

class BitshuffleStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "bitshuffle"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::BitshuffleStage>();
        s->setElementWidth(p.element_width);
        s->setBlockSize(p.block_size);
        return s;
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set(std::format("fzgpumodules:{}:element_width", sid), p.element_width);
        opts.set(std::format("fzgpumodules:{}:block_size",    sid), p.block_size);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(!params_.contains(sid)) params_[sid] = BitshuffleParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get(std::format("fzgpumodules:{}:element_width", sid), &p.element_width);
        opts.get(std::format("fzgpumodules:{}:block_size",    sid), &p.block_size);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set(std::format("fzgpumodules:{}:element_width", sid),
            std::string("Element width in bytes for bit-matrix transpose: 1, 2, 4, or 8 (default 4). "
            "Must match the actual element type of the incoming data."));
        opts.set(std::format("fzgpumodules:{}:block_size", sid),
            std::string("Chunk size in bytes (default 16384; must be multiple of 1024*element_width)"));
    }

private:
    std::map<std::string, BitshuffleParams> params_;
    const BitshuffleParams defaults_{};

    const BitshuffleParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
