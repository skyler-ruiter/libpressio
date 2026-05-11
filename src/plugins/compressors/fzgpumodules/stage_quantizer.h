#pragma once
#include <limits>
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct QuantizerParams {
    int   quant_radius      = 32768;
    float outlier_capacity  = 0.2f;
    bool  zigzag_codes      = false;
    float value_base        = 0.0f;   // 0 = auto-scan; >0 = skip NOA scan
    float outlier_threshold = std::numeric_limits<float>::infinity();  // |x| >= threshold → lossless
    bool  inplace_outliers  = false;  // requires zigzag_codes=true AND sizeof(TCode)==sizeof(TIn)
    bool operator==(const QuantizerParams& o) const {
        return quant_radius == o.quant_radius &&
               outlier_capacity == o.outlier_capacity &&
               zigzag_codes == o.zigzag_codes &&
               value_base == o.value_base &&
               outlier_threshold == o.outlier_threshold &&
               inplace_outliers == o.inplace_outliers;
    }
    bool operator!=(const QuantizerParams& o) const { return !(*this == o); }
};

class QuantizerStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "quantizer"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& in_t  = (parts.size() > 1) ? parts[1] : "float";
        const std::string& cod_t = (parts.size() > 2) ? parts[2] : "uint16";

        if(in_t == "float"  && cod_t == "uint16") return make<float,  uint16_t>(sid, ctx);
        if(in_t == "float"  && cod_t == "uint32") return make<float,  uint32_t>(sid, ctx);
        if(in_t == "double" && cod_t == "uint16") return make<double, uint16_t>(sid, ctx);
        if(in_t == "double" && cod_t == "uint32") return make<double, uint32_t>(sid, ctx);
        throw std::runtime_error("Unsupported quantizer types: " + in_t + ":" + cod_t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":quant_radius",      p.quant_radius);
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",  p.outlier_capacity);
        opts.set("fzgpumodules:" + sid + ":zigzag_codes",      p.zigzag_codes);
        opts.set("fzgpumodules:" + sid + ":value_base",        p.value_base);
        opts.set("fzgpumodules:" + sid + ":outlier_threshold", p.outlier_threshold);
        opts.set("fzgpumodules:" + sid + ":inplace_outliers",  p.inplace_outliers);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = QuantizerParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":quant_radius",      &p.quant_radius);
        opts.get("fzgpumodules:" + sid + ":outlier_capacity",  &p.outlier_capacity);
        opts.get("fzgpumodules:" + sid + ":zigzag_codes",      &p.zigzag_codes);
        opts.get("fzgpumodules:" + sid + ":value_base",        &p.value_base);
        opts.get("fzgpumodules:" + sid + ":outlier_threshold", &p.outlier_threshold);
        opts.get("fzgpumodules:" + sid + ":inplace_outliers",  &p.inplace_outliers);
        return p != old;
    }

    std::string outlier_indices_key(const std::string& /*token*/) const override {
        return "outlier_idxs";
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":quant_radius",
            std::string("Quantization radius (bin count / 2)"));
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",
            std::string("Outlier buffer reserve as fraction of total elements"));
        opts.set("fzgpumodules:" + sid + ":zigzag_codes",
            std::string("Zigzag-encode codes before downstream storage"));
        opts.set("fzgpumodules:" + sid + ":value_base",
            std::string("Pre-computed value_range (NOA) or max(|data|) (REL) to skip NOA data scan; 0 = auto"));
        opts.set("fzgpumodules:" + sid + ":outlier_threshold",
            std::string("ABS/NOA: |x| >= threshold stored losslessly (default: infinity = disabled)"));
        opts.set("fzgpumodules:" + sid + ":inplace_outliers",
            std::string("Encode outliers in-place in codes array; requires zigzag_codes=true "
            "and sizeof(TCode)==sizeof(TIn) (default: false)"));
    }

private:
    std::map<std::string, QuantizerParams> params_;
    const QuantizerParams defaults_{};

    const QuantizerParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename TIn, typename TCode>
    fz::Stage* make(const std::string& sid, const StageContext& ctx) {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::QuantizerStage<TIn, TCode>>();
        s->setErrorBound(static_cast<TIn>(ctx.eb));
        s->setErrorBoundMode(ctx.eb_mode);
        s->setQuantRadius(p.quant_radius);
        s->setOutlierCapacity(p.outlier_capacity);
        s->setZigzagCodes(p.zigzag_codes);
        if(p.value_base > 0.0f) s->setValueBase(p.value_base);
        s->setOutlierThreshold(p.outlier_threshold);
        s->setInplaceOutliers(p.inplace_outliers);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns
