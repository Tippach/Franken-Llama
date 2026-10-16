#include "draft-length.h"

#include <algorithm>
#include <cmath>

common_draft_length_controller::common_draft_length_controller(const common_draft_length_config & cfg)
    : cfg_(cfg) {
    cfg_.max_width     = std::max(1, std::min(cfg_.max_width, common_draft_length_config::kMaxWidth));
    cfg_.default_width = std::min(std::max(cfg_.default_width, cfg_.allow_zero ? 0 : 1), cfg_.max_width);
    cfg_.block         = std::max(1, cfg_.block);
    cfg_.min_samples   = std::max(1, cfg_.min_samples);
    cfg_.alpha         = std::min(std::max(cfg_.alpha, 0.01f), 1.0f);

    // the prior is the mean acceptance of a drafting MTP head, so a conversation that has not been
    // measured yet starts optimistic and shortens on evidence rather than on a pessimistic default
    for (auto & p : p_) {
        p = 0.75f;
    }
    width_ = cfg_.default_width;
    left_  = 0;
}

float common_draft_length_controller::expected_tokens(int width) const {
    // the token the target always commits, plus each draft position that survives to be accepted
    float expected = 1.0f;
    float survival = 1.0f;
    const int w = std::min(width, cfg_.max_width);
    for (int i = 0; i < w; ++i) {
        survival *= p_[i];
        expected += survival;
    }
    return expected;
}

float common_draft_length_controller::cost_ratio(int width, int64_t ctx_tokens) const {
    const int w = std::max(0, std::min(width, common_draft_length_config::kMaxWidth));
    float ratio = cfg_.cost_ratio[w];
    if (w > 0 && cfg_.ctx_slope > 0.0f && ctx_tokens > cfg_.ctx_ref) {
        // the extra draft cells cost more the longer the prompt: issue #18 measured the pass that
        // drafts one more token going from 1.14x at short context to 1.30x at 85K. Only the
        // marginal part of the cost scales with the context; the read of the weights does not.
        const float excess = (float) (ctx_tokens - cfg_.ctx_ref) / (float) cfg_.ctx_ref;
        ratio = 1.0f + (ratio - 1.0f) * (1.0f + cfg_.ctx_slope * std::min(excess, cfg_.ctx_cap));
    }
    return ratio;
}

int common_draft_length_controller::best_width(int64_t ctx_tokens) const {
    // plain decode is the baseline the drafts have to beat: 1 token per pass at cost(0)
    float best_score = 1.0f / cost_ratio(0, ctx_tokens);
    int   best       = 0;
    bool  measured   = false;

    for (int w = 1; w <= cfg_.max_width; ++w) {
        // a width is eligible only once its LAST position has been observed enough: position i is
        // observed only when the draft reached it, so a width never proposed has no evidence and
        // must neither win nor lose on a stale prior. Higher widths stop being considered here.
        if (n_[w - 1] < cfg_.min_samples) {
            break;
        }
        measured = true;
        const float score = expected_tokens(w) / cost_ratio(w, ctx_tokens);
        if (score > best_score * (1.0f + cfg_.margin)) {
            best_score = score;
            best       = w;
        }
    }
    // nothing has been measured yet: draft the configured default and gather samples
    if (!measured) {
        return cfg_.default_width;
    }
    if (best == 0 && !cfg_.allow_zero) {
        best = 1;
    }
    return best;
}

int common_draft_length_controller::take(int64_t ctx_tokens) {
    if (left_ <= 0) {
        ++block_;
        int w = best_width(ctx_tokens);
        // one block in `probe_every_blocks` drafts one token further than the objective wants:
        // without it a width that stops being chosen stops being observed, and the controller can
        // never grow back when the text turns easy to draft (issue #18's staleness, in miniature)
        probe_left_ = 0;
        if (cfg_.probe_every_blocks > 0 && w < cfg_.max_width && block_ % cfg_.probe_every_blocks == 0) {
            probe_left_ = cfg_.block;
            ++w;
            counters_.probes++;
        }
        width_      = w;
        left_       = cfg_.block;
        last_width_ = w;
        counters_.blocks++;
        if (width_ == 0) {
            counters_.zero_blocks++;
        }
    }
    left_--;
    if (probe_left_ > 0) {
        probe_left_--;
    }
    counters_.used[std::min(width_, common_draft_length_config::kMaxWidth)]++;
    return width_;
}

void common_draft_length_controller::observe(size_t accepted, size_t drafted) {
    if (drafted == 0) {
        return;
    }
    const size_t w = std::min(drafted, (size_t) cfg_.max_width);
    const size_t a = std::min(accepted, w);
    // position i is an observation iff the draft reached it: i < a accepted, i == a rejected,
    // i > a never verified and therefore not an observation at all (their "censored" tail)
    for (size_t i = 0; i < a; ++i) {
        p_[i] += cfg_.alpha * (1.0f - p_[i]);
        if (n_[i] < (1 << 20)) {
            n_[i]++;
        }
    }
    if (a < w) {
        p_[a] += cfg_.alpha * (0.0f - p_[a]);
        if (n_[a] < (1 << 20)) {
            n_[a]++;
        }
    }
    counters_.detections++;
}

void common_draft_length_controller::boundary() {
    // keep the estimates but cap their weight, so a conversation's next request starts warm
    // without being stuck on the previous request's kind of text
    for (auto & n : n_) {
        n = std::min(n, cfg_.min_samples * 2);
    }
    left_         = 0;
    probe_left_   = 0;
    last_width_   = -1;
}
