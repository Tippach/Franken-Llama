#pragma once

// strixllama: draft-length controller - choose how many tokens the MTP drafter should propose,
// per conversation, from what the draft has actually been accepting on this text.
//
// Adapted from HaloForge Labs "Halowin Bubo" (MIT), src/models/qwen38_flash_next/mtp_policy.hpp
// (MtpLengthController) and mtp_costs.hpp. It is an idea-level port, not a cherry-pick: their base
// is gufo, ours is llama.cpp, and their cost constants do not transfer (see below).
//
// WHY. Speculative decoding is a gamble: every drafted token costs a slot in the verify batch, and
// pays back only when it is accepted. Whether it pays depends on the text. Measured on this box
// (gfx1151, shipping config: draft-mtp, --spec-draft-vocab-latin, p_min 0.30, temp 1.0):
//
//   prose  acceptance per position = (0.815, 0.576, 0.359)  mean draft len 2.75
//   CJK    acceptance per position = (0.410, 0.025, 0.012)  mean draft len 1.45
//
// On prose, drafting 3 tokens is worth it. On CJK, positions 2 and 3 land 2.5% and 1.2% of the
// time and the extra verify cells make the pass 16% slower than plain decode: 43.58 ms/token with
// the draft against 37.51 ms/token with no draft at all. The shipping configuration is a fixed
// --spec-draft-n-max 3 for every kind of text, so CJK-like traffic pays that 16% permanently.
//
// WHAT MAKES IT WORK. The expected tokens a pass commits are
//
//   E(w) = 1 + sum over i<=w of P(at least i draft tokens accepted)
//
// (the leading 1 is the token the target always commits), and the width to use is the one that
// maximises E(w)/cost(w). P is estimated per conversation by an exponential moving average of what
// its own drafts have accepted, so a conversation that switches from English to Chinese converges
// to the short drafts on its own.
//
// WHY THEIR CONSTANTS DO NOT TRANSFER. mtp_costs.hpp gives C1 widths 0..3 as 38.5, 47.3, 56.1,
// 65.7 ms, i.e. cost(1)/cost(0) = 1.229. Measured here, cost(1)/cost(0) = 1.361 (see
// harness/_draft_cost.ps1: 47.02, 64.0, 77.1, 87.7 ms per pass for widths 0..3). Their absolute
// cost at width 3 does transfer (implied 1.675 here against their 1.706), but their MARGINAL for
// the first drafted token is ~15% too cheap for this hardware, and a cost table that underprices
// the first draft token over-drafts. The cost table below is ours, measured on this box.
//
// WHY A STATIC COST TABLE. Cost per pass depends on the verify shape, which depends on the width,
// but it does not depend on the text - and it is expensive to measure live: a width that is not
// chosen stops being timed, which is what made the previous attempt (issue #18) go stale and stick.
// The table is measured once, offline, and only its context scaling is estimated.
//
// WHAT THE OBJECTIVE PREDICTS, and the measurement agrees. Note llama.cpp reports the acceptance
// rate per position UNCONDITIONALLY (mean len = 1 + sum of them), so the objective's conditional
// probabilities are ratios of those. On the prose profile (0.815, 0.576, 0.359 -> conditional
// 0.815, 0.707, 0.623) the scores E(w)/cost(w) are 1.334 (w=1), 1.459 (w=2), 1.475 (w=3) against
// 1.000 for plain decode. w=3 is the raw argmax but only 1.1% ahead of w=2, inside the 2% margin,
// so the controller picks 2 - and the measured pass times agree: 31.79 tokens/s at width 2 against
// 30.91 at width 3. The shipping --spec-draft-n-max 3 is one token past the optimum on prose.
// On the CJK profile (0.410, 0.025, 0.012 -> conditional 0.410, 0.061, 0.48) the scores are 1.036
// (w=1) and 0.876 (w=2), so it drafts a single token - and the measured width-1 run came in at
// 37.45 ms/token against 37.51 with no draft and 43.58 at width 3. Both predictions land within
// the noise of the measurement, which is the evidence that the objective is the right one here.
//
// OUTPUT COMPATIBILITY. Changing the width changes the target verify batch, which changes the
// forward pass's reduction order by ulps, which can flip one greedy argmax and cascade into
// different text. Measured: width 0 / 1 / 2 / 3 produced 1253 / 1248 / 1248 / 1229 chars on the
// same greedy prompt. So this is NOT output-neutral and must not be gated on byte-identity
// against a fixed-width run; it is gated on quality (canonical PPL) and on throughput.
//
// Cost when the text is stable: a handful of float multiplies per pass on the host, no GPU work.

#include <array>
#include <cstddef>
#include <cstdint>

struct common_draft_length_config {
    // highest width considered; the shipping --spec-draft-n-max is 3
    static constexpr int kMaxWidth = 8;

    int   max_width = 3;
    // the width used until the acceptance estimates have enough samples to choose
    int   default_width = 3;
    // an arm needs this many observed passes before it can win
    int   min_samples = 8;
    // a chosen width holds for this many passes: each switch is another verify shape, and the
    // passes right after a switch pay for building it (measured ~30 ms dearer, issue #18)
    int   block = 32;
    // passes discarded at the start of a block after the width changed
    int   untimed = 2;
    // the width must beat plain decode by this fraction to be worth the drafts
    float margin = 0.02f;
    // EMA weight of the newest observation
    float alpha = 0.25f;
    // one block in this many forces one width further, so a width no longer chosen keeps being
    // observed and the controller can grow back when the text turns easy to draft
    int   probe_every_blocks = 4;
    // allow the controller to draft nothing at all when no width beats plain decode
    bool  allow_zero = true;

    // cost(w)/cost(0), measured on this box at ctx 131072, greedy, prose (harness/_draft_cost.ps1,
    // 2026-10-13): pass cost 47.02, 64.0, 77.1, 87.7 ms for widths 0..3. Only the ratio matters:
    // the objective compares E(w)/cost(w) against 1/cost(0), so cost(0) cancels.
    std::array<float, kMaxWidth + 1> cost_ratio = { 1.000f, 1.361f, 1.639f, 1.864f,
                                                    2.05f, 2.22f, 2.37f, 2.50f, 2.62f };
    // knob: how much the marginal cost of a draft token grows with the context. Left at 0: the
    // two measurements on 2026-10-13 say the ratio is flat in depth - cost(1)/cost(0) = 1.361 on
    // prose at a ~60k-token prompt and 1.378 on CJK at a ~50-token prompt. Issue #18 had inferred
    // a rise (1.14 short -> 1.30 at 85K), but that was the 2->3 step read from a stale table rather
    // than measured, and the fresh numbers do not support it. Kept as a knob in case a depth sweep
    // later disagrees; cost_ratio() only applies it above ctx_ref.
    float ctx_slope = 0.0f;
    int   ctx_ref   = 85000;
    float ctx_cap   = 1.5f;
};

struct common_draft_length_counters {
    int64_t detections = 0;   // passes observed
    int64_t blocks       = 0; // widths chosen
    int64_t probes       = 0; // blocks spent one width further than chosen
    int64_t zero_blocks  = 0; // blocks that drafted nothing
    std::array<int64_t, common_draft_length_config::kMaxWidth + 1> used = {};
};

// Pure per-conversation controller: it sees a pass's outcome and the context length, and returns
// the draft width. No llama state, so it is unit-testable (tests/test-draft-length.cpp).
//
// The state is per conversation on purpose: acceptance is a property of the text being generated,
// and a chat that turns from code to prose should converge to prose's answer without dragging
// code's history along (issue #18 hit exactly this and caps the inherited weight at reset).
class common_draft_length_controller {
public:
    explicit common_draft_length_controller(const common_draft_length_config & cfg = {});

    // the width to draft for this pass; holds the previous choice inside a block
    int take(int64_t ctx_tokens);

    // a pass finished: `drafted` were proposed, `accepted` of them were committed
    void observe(size_t accepted, size_t drafted);

    // a new conversation: keep the estimates at a small weight so the next request starts warm
    // without inheriting the last request's answer
    void boundary();

    int  width() const { return width_; }
    bool probing() const { return probe_left_ > 0; }

    const common_draft_length_config &      config()  const { return cfg_; }
    common_draft_length_counters            counters() const { return counters_; }

    // P(at least i+1 draft tokens accepted), the estimate the objective is built from
    float acceptance(int i) const { return i >= 0 && i < cfg_.max_width ? p_[i] : 0.0f; }
    int   samples(int i)    const { return i >= 0 && i < cfg_.max_width ? n_[i] : 0; }

    // expected tokens committed by a pass that drafts `width` tokens, from the current estimates
    float expected_tokens(int width) const;

    // cost(width)/cost(0) at a given context length
    float cost_ratio(int width, int64_t ctx_tokens) const;

    // the width the objective prefers right now, ignoring blocks and probes
    int best_width(int64_t ctx_tokens) const;

private:
    common_draft_length_config cfg_;
    // p_[i] estimates P(>= i+1 draft tokens accepted); the prior is the mean acceptance of a
    // drafting MTP head, so an unmeasured conversation starts optimistic and shortens only on
    // evidence. Their state uses the equivalent 1.5/(1.5+0.5).
    std::array<float, common_draft_length_config::kMaxWidth> p_ = {};
    std::array<int,    common_draft_length_config::kMaxWidth> n_ = {};

    int width_      = 0;
    int left_       = 0;   // passes this width still holds
    int probe_left_ = 0;   // passes forced one width further
    int block_      = 0;
    int last_width_ = -1;

    common_draft_length_counters counters_;
};
