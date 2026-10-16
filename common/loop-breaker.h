#pragma once

// strixllama: loop breaker - detect and break token loops in generated text.
//
// Ported from HaloForge Labs "Halowin Bubo" (MIT), src/core/behaviour.hpp:136 (LoopBreaker),
// which is itself a value-state port of their production common/loop-breaker.h. The detector is
// pure: no llama context, no model, no GPU - it sees committed token ids and nothing else, which
// makes it unit-testable (tests/test-loop-breaker.cpp).
//
// Why: long reasoning generations fail in two ways that a sampler cannot see coming.
//   - inside <think>: the model repeats a block until the context limit. The user gets no answer.
//   - inside the answer: the model repeats a block and never emits the end-of-turn token.
// Both were reproduced against gufo/Bubo upstream: 14 of 15 long answers stopped mid-answer without
// their end-of-turn fix. The fix is a policy on top of sampling, not a model change.
//
// Two-stage response, deliberately conservative:
//   1. NUDGE  - raise the effective temperature for a while. A loop can often be broken without
//               touching the content, so try that first. No-op at temp == 0 (greedy has no
//               temperature to raise); the terminal action then does the work.
//   2. terminal - if the loop is still there after the nudge expires, force the thinking block
//               closed (FORCE_THINK) or force end-of-generation (STOP_ANSWER).
//
// The detector finds a period p such that the tail of the stream is (almost) a repetition of one
// block of length p, and requires BOTH a minimum number of repeats and a minimum number of tokens,
// per channel. Answer thresholds are much stricter than thinking thresholds: repeated structure in
// JSON/code inside an answer is legitimate, so a short period must repeat a very long time before
// the answer is cut off.
//
// Cost when nothing loops: O(max_period) integer compares per token, on host only. No effect on
// decode bandwidth, no effect on the GPU.

#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// pure detector (no llama types beyond the token id)
// ---------------------------------------------------------------------------

struct common_loop_breaker_config {
    int think_repeats      = 16;  // min whole repeats of the block inside <think>
    int think_tokens       = 128; // min tokens covered by the repetition inside <think>
    int answer_repeats     = 32;  // min whole repeats in the answer (stricter: JSON/code repeat legitimately)
    int answer_tokens      = 512; // min tokens covered by the repetition in the answer
    int think_max_period   = 128; // longest block considered a repeating unit inside <think>
    int answer_max_period  = 16;  // longest block considered a repeating unit in the answer
    int nudge_tokens       = 16;  // tokens to keep the nudge active after a detection
    int watch_tokens       = 256; // keep watching this long past the nudge for a relapse
};

struct common_loop_breaker_state {
    enum channel { ANSWER = 0, THINK = 1 };
    enum action  { NONE = 0, NUDGE = 1, FORCE_THINK = 2, STOP_ANSWER = 3 };

    static constexpr int kMaxPeriod = 128;

    struct event {
        action  what    = NONE;
        channel where   = ANSWER;
        int     period  = 0;
        int     repeats = 0;
    };

    common_loop_breaker_config cfg;

    // running "how many tokens in a row has this period matched" per period, and the last
    // kMaxPeriod tokens so a period can be extended without rescanning the whole stream
    std::array<int32_t,     kMaxPeriod> tail    {};
    std::array<uint32_t,    kMaxPeriod> matching {};
    uint64_t count = 0;

    channel current        = ANSWER;
    int     nudge_left     = 0;
    int     watch_left     = 0;
    int     nudges_in_span = 0;
    bool    terminal       = false;

    // counters, for logging/gates - never read by the decision path
    uint64_t detections      = 0;
    uint64_t nudges          = 0;
    uint64_t forced_thinks   = 0;
    uint64_t stopped_answers = 0;

    void reset(void);

    // start a fresh span (new thinking block / new answer): forget the tail and the nudge
    void boundary(channel next);

    bool nudging(void) const { return nudge_left > 0; }

    // feed one committed token; returns the action to take, NONE if the stream looks fine
    event accept(int32_t token, channel where);
};

// ---------------------------------------------------------------------------
// sampler wrapper: channel tracking + apply() (nudge scaling / forced tokens)
// ---------------------------------------------------------------------------

// Build a loop-breaker sampler.
//   think_start / think_end - token sequences marking the reasoning span. If think_start is empty
//     the sampler has no thinking channel and only guards the answer.
//   forced_close            - token sequence that ends a thinking span (normally the budget message +
//     </think>). Informational for the sampler: the close is carried out by the caller through the
//     reasoning-budget sampler, which already owns this sequence. May be empty.
//   eog                     - end-of-generation token. Also informational: the sampler cannot emit a
//     token that is not among its candidates (see common_loop_breaker_take_request), so ending the
//     generation is the caller's job too. Pass llama_vocab_eos()/eot(), or < 0 to say "none".
//   temp                    - the request temperature, used only to convert "temperature + nudge"
//                             into a logit scale factor. <= 0 (greedy) disables the nudge.
//   nudge_add               - how much to add to the effective temperature while nudging.
//
// The sampler therefore does exactly one thing to the logits itself: the nudge. Everything terminal is
// a request for the caller. A caller that never polls take_request() still gets the nudge, and gets
// detection + logging, but not the cutoff - llama-server does poll it (see process_token).
struct llama_sampler * common_loop_breaker_init(
        const std::vector<llama_token> & think_start,
        const std::vector<llama_token> & think_end,
        const std::vector<llama_token> & forced_close,
        llama_token                      eog,
        float                            temp,
        const common_loop_breaker_config & cfg = {},
        float                              nudge_add = 0.5f);

// Take a pending terminal request, clearing it. Returns:
//   0 - nothing pending
//   1 - close the thinking span: the caller drives the reasoning-budget sampler, which owns the
//       forced end sequence and can actually emit it
//   2 - end the generation: the caller sets its own stop flag
// The sampler cannot do either itself: it is only allowed to choose among the candidates it is given,
// and end-of-turn / </think> are frequently not among them (the server pre-filters the candidate set).
// Masking logits to force them silently changed nothing - measured, not assumed.
int common_loop_breaker_take_request(struct llama_sampler * smpl);

// current channel the sampler is watching (for logging/tests)
common_loop_breaker_state::channel common_loop_breaker_get_state(const struct llama_sampler * smpl);

// Tell the sampler to stand down for the next apply(). Used when a strict grammar/constraint is
// deciding this position: whatever a schema forces to repeat is not a model loop. One flag per
// position - the caller sets it before each apply. Mirrors the upstream breaker, which is skipped
// while a constraint is active.
void common_loop_breaker_set_exempt(struct llama_sampler * smpl, bool exempt);

// diagnostics: number of detections / nudges / forced thinks / stopped answers
void common_loop_breaker_get_counters(const struct llama_sampler * smpl,
                                      uint64_t * detections,
                                      uint64_t * nudges,
                                      uint64_t * forced_thinks,
                                      uint64_t * stopped_answers);
