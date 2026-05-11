#pragma once
#include <compare>
#include <limits>
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct LorenzoParams {
    int   quant_radius     = 32768;
    float outlier_capacity = 0.2f;
    bool  zigzag_codes     = false;
    float value_base       = 0.0f;  // 0 = auto-scan; >0 = skip scan (NOA: value_range, REL: max|data|)
    auto operator<=>(const LorenzoParams&) const = default;
};

class LorenzoStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "lorenzo"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& in_t = (parts.size() > 1) ? parts[1] : "float";

        // Integer (lossless) Lorenzo: no configurable params
        if(in_t == "int8")  return ctx.pipeline.addStage<fz::LorenzoStage<int8_t>>();
        if(in_t == "int16") return ctx.pipeline.addStage<fz::LorenzoStage<int16_t>>();
        if(in_t == "int32") return ctx.pipeline.addStage<fz::LorenzoStage<int32_t>>();
        if(in_t == "int64") return ctx.pipeline.addStage<fz::LorenzoStage<int64_t>>();

        // Quantizing Lorenzo: error-bound-driven
        std::string cod_t = (parts.size() > 2 && !parts[2].empty()) ? parts[2] : "uint16";
        if(in_t == "float"  && cod_t == "uint16") return make<float,  uint16_t>(sid, ctx);
        if(in_t == "float"  && cod_t == "uint8")  return make<float,  uint8_t> (sid, ctx);
        if(in_t == "double" && cod_t == "uint16") return make<double, uint16_t>(sid, ctx);
        if(in_t == "double" && cod_t == "uint32") return make<double, uint32_t>(sid, ctx);
        throw std::runtime_error("Unsupported lorenzo types: " + in_t + ":" + cod_t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& token) const override {
        if(is_int(token)) return;
        const auto& p = get_params(sid);
        opts.set(std::format("fzgpumodules:{}:quant_radius",     sid), p.quant_radius);
        opts.set(std::format("fzgpumodules:{}:outlier_capacity", sid), p.outlier_capacity);
        opts.set(std::format("fzgpumodules:{}:zigzag_codes",     sid), p.zigzag_codes);
        opts.set(std::format("fzgpumodules:{}:value_base",       sid), p.value_base);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     token) override {
        if(is_int(token)) return false;
        if(!params_.contains(sid)) params_[sid] = LorenzoParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get(std::format("fzgpumodules:{}:quant_radius",     sid), &p.quant_radius);
        opts.get(std::format("fzgpumodules:{}:outlier_capacity", sid), &p.outlier_capacity);
        opts.get(std::format("fzgpumodules:{}:zigzag_codes",     sid), &p.zigzag_codes);
        opts.get(std::format("fzgpumodules:{}:value_base",       sid), &p.value_base);
        return p != old;
    }

    std::string outlier_indices_key(const std::string& token) const override {
        return is_int(token) ? "" : "outlier_indices";
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& token) const override {
        if(is_int(token)) return;
        opts.set(std::format("fzgpumodules:{}:quant_radius",     sid),
            std::string("Quantization radius (bin count / 2)"));
        opts.set(std::format("fzgpumodules:{}:outlier_capacity", sid),
            std::string("Outlier buffer reserve as fraction of total elements"));
        opts.set(std::format("fzgpumodules:{}:zigzag_codes",     sid),
            std::string("Zigzag-encode codes before downstream storage"));
        opts.set(std::format("fzgpumodules:{}:value_base",       sid),
            std::string("Pre-computed value_range (NOA) or max(|data|) (REL) to skip data scan; 0 = auto"));
    }

private:
    std::map<std::string, LorenzoParams> params_;
    const LorenzoParams defaults_{};

    static bool is_int(const std::string& token) {
        auto parts = split_str(token, ':');
        if(parts.size() < 2) return false;
        const auto& t = parts[1];
        return t == "int8" || t == "int16" || t == "int32" || t == "int64";
    }

    const LorenzoParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename TIn, typename TCode>
    fz::Stage* make(const std::string& sid, const StageContext& ctx) {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::LorenzoQuantStage<TIn, TCode>>();
        s->setErrorBound(static_cast<TIn>(ctx.eb));
        s->setErrorBoundMode(ctx.eb_mode);
        s->setQuantRadius(p.quant_radius);
        s->setOutlierCapacity(p.outlier_capacity);
        s->setZigzagCodes(p.zigzag_codes);
        if(p.value_base > 0.0f) s->setValueBase(p.value_base);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
