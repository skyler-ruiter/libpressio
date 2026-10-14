#pragma once
#include "stages/stage_kind.h"
#include "stages/stage_lorenzo.h"
#include "stages/stage_quantizer.h"
#include "stages/stage_diff.h"
#include "stages/stage_rle.h"
#include "stages/stage_zigzag.h"
#include "stages/stage_negabinary.h"
#include "stages/stage_bitshuffle.h"
#include "stages/stage_bitpack.h"
#include "stages/stage_rze.h"
#include "stages/stage_tiled_lorenzo.h"
#include "stages/stage_adaptive_bitpack.h"
#include "stages/stage_rre.h"
#include "stages/stage_rare.h"
#include "stages/stage_raze.h"
#include "stages/stage_clog.h"
#include "stages/stage_hclog.h"
#include "stages/stage_gpulz.h"
#include "stages/stage_tupl.h"
#include "stages/stage_ans.h"
#if LIBPRESSIO_FZGM_HAS_ADM
#include "stages/stage_adm.h"
#endif
#include "stages/stage_bitplane_rze.h"
#if LIBPRESSIO_FZGM_HAS_SZX
#include "stages/stage_szx.h"
#endif
#if LIBPRESSIO_FZGM_HAS_SZP
#include "stages/stage_szp.h"
#endif
#include "stages/stage_huffman.h"
#include "stages/stage_adaptive_lorenzo.h"
#include "stages/stage_log_transform.h"
#include "stages/stage_ginterp.h"
#include "stages/stage_roibin_split.h"
#if LIBPRESSIO_FZGM_HAS_CDF97
#include "stages/stage_cdf97.h"
#endif
#if LIBPRESSIO_FZGM_HAS_SPECK2D
#include "stages/stage_speck2d.h"
#endif
#if LIBPRESSIO_FZGM_HAS_CDF97_OUTLIER_CORRECT
#include "stages/stage_outlier_correct.h"
#endif

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// The catalog and SDK capability guards live here; the plugin uses token lookup.
class StageRegistry {
public:
    StageRegistry() {
        kinds_.push_back(std::make_unique<LorenzoStageKind>());
        kinds_.push_back(std::make_unique<QuantizerStageKind>());
        kinds_.push_back(std::make_unique<DiffStageKind>());
        kinds_.push_back(std::make_unique<RLEStageKind>());
        kinds_.push_back(std::make_unique<ZigzagStageKind>());
        kinds_.push_back(std::make_unique<NegabinaryStageKind>());
        kinds_.push_back(std::make_unique<BitshuffleStageKind>());
        kinds_.push_back(std::make_unique<BitpackStageKind>());
        kinds_.push_back(std::make_unique<RZEStageKind>());
        kinds_.push_back(std::make_unique<TiledLorenzoStageKind>());
        kinds_.push_back(std::make_unique<AdaptiveBitpackStageKind>());
        kinds_.push_back(std::make_unique<RREStageKind>());
        kinds_.push_back(std::make_unique<RAREStageKind>());
        kinds_.push_back(std::make_unique<RAZEStageKind>());
        kinds_.push_back(std::make_unique<CLOGStageKind>());
        kinds_.push_back(std::make_unique<HCLOGStageKind>());
        kinds_.push_back(std::make_unique<GPULZStageKind>());
        kinds_.push_back(std::make_unique<TUPLStageKind>());
        kinds_.push_back(std::make_unique<ANSStageKind>());
#if LIBPRESSIO_FZGM_HAS_ADM
        kinds_.push_back(std::make_unique<ADMStageKind>());
#endif
        kinds_.push_back(std::make_unique<BitplaneRZEStageKind>());
#if LIBPRESSIO_FZGM_HAS_SZX
        kinds_.push_back(std::make_unique<SZxStageKind>());
#endif
#if LIBPRESSIO_FZGM_HAS_SZP
        kinds_.push_back(std::make_unique<SZpStageKind>());
#endif
        kinds_.push_back(std::make_unique<HuffmanStageKind>());
        kinds_.push_back(std::make_unique<AdaptiveLorenzoStageKind>());
        kinds_.push_back(std::make_unique<LogTransformStageKind>());
        kinds_.push_back(std::make_unique<GInterpStageKind>());
        kinds_.push_back(std::make_unique<ROIBinSplitStageKind>());
#if LIBPRESSIO_FZGM_HAS_CDF97
        kinds_.push_back(std::make_unique<Cdf97StageKind>());
#endif
#if LIBPRESSIO_FZGM_HAS_SPECK2D
        kinds_.push_back(std::make_unique<Speck2DStageKind>());
#endif
#if LIBPRESSIO_FZGM_HAS_CDF97_OUTLIER_CORRECT
        kinds_.push_back(std::make_unique<Cdf97OutlierCorrectStageKind>());
#endif
    }

    StageRegistry(const StageRegistry& other) {
        kinds_.reserve(other.kinds_.size());
        for(const auto& kind : other.kinds_) kinds_.push_back(kind->clone());
    }
    StageRegistry(StageRegistry&&) = default;
    StageRegistry& operator=(StageRegistry&&) = default;
    StageRegistry& operator=(const StageRegistry& other) {
        StageRegistry copy(other);
        swap(copy);
        return *this;
    }
    void swap(StageRegistry& other) noexcept { kinds_.swap(other.kinds_); }

    StageKind* find(std::string_view token) {
        const auto kind = token_kind(token);
        for(auto& entry : kinds_) if(entry->matches(kind)) return entry.get();
        return nullptr;
    }
    const StageKind* find(std::string_view token) const {
        const auto kind = token_kind(token);
        for(const auto& entry : kinds_) if(entry->matches(kind)) return entry.get();
        return nullptr;
    }

private:
    std::vector<std::unique_ptr<StageKind>> kinds_;
};

}}}
