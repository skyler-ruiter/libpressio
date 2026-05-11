#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct RZEParams {
    int chunk_size = 16384;  // bytes; must be multiple of 4096
    int levels     = 4;      // recursion depth 1–4
    bool operator==(const RZEParams& o) const {
        return chunk_size == o.chunk_size && levels == o.levels;
    }
    bool operator!=(const RZEParams& o) const { return !(*this == o); }
};

class RZEStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "rze"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::RZEStage>();
        s->setChunkSize(p.chunk_size);
        s->setLevels(p.levels);
        return s;
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":chunk_size", p.chunk_size);
        opts.set("fzgpumodules:" + sid + ":levels",     p.levels);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = RZEParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":chunk_size", &p.chunk_size);
        opts.get("fzgpumodules:" + sid + ":levels",     &p.levels);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":chunk_size",
            std::string("RZE chunk size in bytes (default 16384; must be multiple of 4096)"));
        opts.set("fzgpumodules:" + sid + ":levels",
            std::string("RZE recursion depth 1-4 (default 4)"));
    }

private:
    std::map<std::string, RZEParams> params_;
    const RZEParams defaults_{};

    const RZEParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
