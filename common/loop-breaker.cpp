#include "loop-breaker.h"

#include "log.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// strixllama: TEST-ONLY fault injection. STRIXLLAMA_LB_FAULT=N makes the sampler emit STOP_ANSWER
// once, after N answer-channel tokens, whatever the text is. It exists so the wiring - armed sampler
// -> action -> forced EOG -> the server actually ending generation - can be proven on a live server
// without depending on a model spontaneously falling into a loop (which is exactly the thing being
// fixed, so it cannot be the thing the test waits for). The detector itself is covered by
// tests/test-loop-breaker.cpp; this hook never fires unless the variable is set to N > 0.
//
// NOTE the name: it deliberately does NOT start with LLAMA_ / GGML_ / STRIX_. The server rebuilds its
// environment at startup and DROPS every variable under those prefixes that is not in the baked gate
// list (tools/server/main.cpp), so an LLAMA_-prefixed hook would be silently dead in the shipping
// binary. STRIXLLAMA_ is outside all three prefixes and survives.
static int loop_breaker_fault_n(void) {
    static const int n = [] {
        const char * e = getenv("STRIXLLAMA_LB_FAULT");
        return (e != nullptr && atoi(e) > 0) ? atoi(e) : 0;
    }();
    return n;
}

// ---------------------------------------------------------------------------
// pure detector
// ---------------------------------------------------------------------------

void common_loop_breaker_state::reset(void) {
    const common_loop_breaker_config saved = cfg;

    *this = common_loop_breaker_state {};
    cfg = saved;
}

void common_loop_breaker_state::boundary(channel next) {
    tail.fill(0);
    matching.fill(0);
    count          = 0;
    current        = next;
    nudge_left     = 0;
    watch_left     = 0;
    nudges_in_span = 0;
    terminal       = false;
}

common_loop_breaker_state::event common_loop_breaker_state::accept(int32_t token, channel where) {
    if (where != current) {
        boundary(where);
    }

    if (terminal) {
        return {};
    }

    if (nudge_left > 0) {
        --nudge_left;
    }
    if (watch_left > 0) {
        --watch_left;
    }

    // incremental period detection: for each candidate period p, matching[p-1] counts how many
    // tokens in a row the token p positions back has equalled the current one. a stream that is a
    // clean repetition of a block of length p drives matching[p-1] upwards; anything else resets it.
    int period  = 0;
    int repeats = 0;

    const int min_repeats = where == THINK ? cfg.think_repeats  : cfg.answer_repeats;
    const int min_tokens  = where == THINK ? cfg.think_tokens   : cfg.answer_tokens;
    const int max_period  = std::clamp(where == THINK ? cfg.think_max_period : cfg.answer_max_period,
                                       0, kMaxPeriod);

    for (int p = 1; p <= max_period; ++p) {
        auto & run = matching[p - 1];

        if (count >= (uint64_t) p && tail[(count - p) % tail.size()] == token) {
            if (run != std::numeric_limits<uint32_t>::max()) {
                ++run;
            }
        } else {
            run = 0;
        }

        const uint64_t pure = (uint64_t) run + p;

        if (!period && run >= (uint32_t) p && pure >= (uint64_t) min_tokens
                && pure / p >= (uint64_t) min_repeats) {
            period  = p;
            repeats = (int) std::min<uint64_t>(pure / p, (uint64_t) std::numeric_limits<int32_t>::max());
        }
    }

    tail[count % tail.size()] = token;
    ++count;

    event found {};

    if (period && nudges_in_span == 0) {
        // first hit in this span: try the gentle fix and keep watching
        found = { NUDGE, where, period, repeats };

        nudge_left     = cfg.nudge_tokens;
        watch_left     = cfg.nudge_tokens + cfg.watch_tokens;
        nudges_in_span++;

        ++detections;
        ++nudges;
    } else if (period && nudges_in_span >= 1 && !nudging()) {
        // still looping after the nudge wore off: cut the span short
        found = { where == THINK ? FORCE_THINK : STOP_ANSWER, where, period, repeats };

        nudge_left = 0;
        watch_left = 0;
        terminal   = true;

        ++detections;
        if (where == THINK) {
            ++forced_thinks;
        } else {
            ++stopped_answers;
        }
    }

    return found;
}

// ---------------------------------------------------------------------------
// sampler wrapper
// ---------------------------------------------------------------------------

struct common_loop_breaker_ctx {
    common_loop_breaker_state st;

    std::vector<llama_token> think_start;
    std::vector<llama_token> think_end;
    std::vector<llama_token> forced_close;   // what to force to end a thinking span
    llama_token              eog = LLAMA_TOKEN_NULL;

    float temp       = 0.0f;                 // request temperature, for the nudge scale factor
    float nudge_add  = 0.5f;                 // "temperature + 0.5"

    // reasoning-span tracking: the last max(start,end) committed tokens, matched as suffixes
    std::vector<llama_token> match_tail;

    bool   thinking  = false;
    bool   exempt    = false;   // a strict grammar is deciding this position - stand down
    bool   fault_fired = false; // see loop_breaker_fault_n()
    size_t max_match = 0;

    // strixllama: terminal actions are REQUESTS, not logit masks. The server can hand the CPU sampler
    // chain a pre-filtered candidate set (measured: cur_p arrived with a single already-chosen token),
    // and a sampler cannot emit a token that is not in its candidates - masking to force </think> or
    // end-of-turn just logged "forced token is not in the candidate set" and changed nothing. So the
    // caller polls these requests and uses the mechanism that can actually act: the reasoning-budget
    // force for a thinking span, and its own stop flag to end a generation.
    bool ask_close_think = false;
    bool ask_stop        = false;

    bool ends_with(const std::vector<llama_token> & seq) const {
        return !seq.empty() && match_tail.size() >= seq.size()
            && std::equal(seq.rbegin(), seq.rend(), match_tail.rbegin());
    }
};

static const char * common_loop_breaker_name(const struct llama_sampler * /*smpl*/) {
    return "loop-breaker";
}

static void common_loop_breaker_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_loop_breaker_ctx *) smpl->ctx;

    // the reasoning span is decided by the marker tokens, not by the loop breaker
    ctx->match_tail.push_back(token);
    if (ctx->max_match == 0) {
        ctx->match_tail.clear();
    } else if (ctx->match_tail.size() > ctx->max_match) {
        ctx->match_tail.erase(ctx->match_tail.begin(), ctx->match_tail.end() - ctx->max_match);
    }

    if (!ctx->thinking && ctx->ends_with(ctx->think_start)) {
        ctx->thinking = true;
        ctx->st.boundary(common_loop_breaker_state::THINK);
        return;
    }

    if (ctx->thinking && ctx->ends_with(ctx->think_end)) {
        ctx->thinking = false;
        ctx->st.boundary(common_loop_breaker_state::ANSWER);
        return;
    }

    // while a close has been requested the detector is not fed: the tokens that end the span are ours,
    // not the model's, and feeding a forced </think> back would let the detector react to its own
    // output.
    if (ctx->ask_close_think || ctx->ask_stop) {
        return;
    }

    const auto ev = ctx->st.accept((int32_t) token,
                                   ctx->thinking ? common_loop_breaker_state::THINK
                                                 : common_loop_breaker_state::ANSWER);

    // test-only: pretend a terminal answer loop happened, to prove the wiring end to end
    const int fault_n = loop_breaker_fault_n();
    if (fault_n > 0 && !ctx->thinking && !ctx->fault_fired && !ctx->ask_stop
            && ctx->st.count >= (uint64_t) fault_n) {
        ctx->fault_fired = true;
        LOG_WRN("%s: FAULT INJECTION (STRIXLLAMA_LB_FAULT=%d) ctx=%p - requesting end of generation\n",
                __func__, fault_n, (void *) ctx);
        ctx->ask_stop = true;
        return;
    }

    if (ev.what == common_loop_breaker_state::NONE) {
        return;
    }

    if (ev.what == common_loop_breaker_state::NUDGE) {
        LOG_INF("%s: loop detected channel=%s period=%d repeats=%d - nudging temperature +%.1f for %d tokens\n",
                __func__, ctx->thinking ? "think" : "answer", ev.period, ev.repeats,
                ctx->nudge_add, ctx->st.cfg.nudge_tokens);
        return;
    }

    if (ev.what == common_loop_breaker_state::FORCE_THINK) {
        LOG_INF("%s: loop persists after nudge channel=think period=%d repeats=%d - requesting that the thinking block be closed\n",
                __func__, ev.period, ev.repeats);
        ctx->ask_close_think = true;
        return;
    }

    if (ev.what == common_loop_breaker_state::STOP_ANSWER) {
        LOG_INF("%s: loop persists after nudge channel=answer period=%d repeats=%d - requesting that generation end\n",
                __func__, ev.period, ev.repeats);
        ctx->ask_stop = true;
        return;
    }
}

static void common_loop_breaker_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_loop_breaker_ctx *) smpl->ctx;

    // apply() only ever adjusts the logits. The terminal actions are not here: a sampler chooses among
    // the candidates it is handed, and </think> / end-of-turn are regularly not among them, so masking
    // to force one was a silent no-op (measured). See common_loop_breaker_take_request.
    if (ctx->st.nudging() && !ctx->exempt && ctx->temp > 0.0f) {
        // this sampler runs after the temperature sampler, which divided by temp. dividing again
        // by (temp + nudge) is the same as multiplying by temp / (temp + nudge).
        const float scale = ctx->temp / (ctx->temp + ctx->nudge_add);

        for (size_t i = 0; i < cur_p->size; ++i) {
            if (std::isfinite(cur_p->data[i].logit)) {
                cur_p->data[i].logit *= scale;
            }
        }
    }
}

static void common_loop_breaker_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_loop_breaker_ctx *) smpl->ctx;

    ctx->st.reset();
    ctx->match_tail.clear();
    ctx->thinking        = false;
    ctx->exempt          = false;
    ctx->ask_close_think = false;
    ctx->ask_stop        = false;
    ctx->fault_fired     = false;
}

static struct llama_sampler * common_loop_breaker_clone(const struct llama_sampler * smpl);

static void common_loop_breaker_free(struct llama_sampler * smpl) {
    delete (common_loop_breaker_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_loop_breaker_i = {
    /* .name              = */ common_loop_breaker_name,
    /* .accept            = */ common_loop_breaker_accept,
    /* .apply             = */ common_loop_breaker_apply,
    /* .reset             = */ common_loop_breaker_reset,
    /* .clone             = */ common_loop_breaker_clone,
    /* .free              = */ common_loop_breaker_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_loop_breaker_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_loop_breaker_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_loop_breaker_i,
        /* .ctx   = */ new common_loop_breaker_ctx(*ctx)
    );
}

struct llama_sampler * common_loop_breaker_init(
        const std::vector<llama_token>   & think_start,
        const std::vector<llama_token>   & think_end,
        const std::vector<llama_token>   & forced_close,
        llama_token                        eog,
        float                              temp,
        const common_loop_breaker_config & cfg,
        float                              nudge_add) {

    auto * ctx = new common_loop_breaker_ctx;

    ctx->st.cfg        = cfg;
    ctx->think_start   = think_start;
    ctx->think_end     = think_end;
    ctx->forced_close  = forced_close;
    ctx->eog           = eog;
    ctx->temp          = temp;
    ctx->nudge_add     = nudge_add;
    ctx->max_match     = std::max(think_start.size(), think_end.size());

    return llama_sampler_init(
        /* .iface = */ &common_loop_breaker_i,
        /* .ctx   = */ ctx
    );
}

int common_loop_breaker_take_request(struct llama_sampler * smpl) {
    if (!smpl) {
        return 0;
    }
    auto * ctx = (common_loop_breaker_ctx *) smpl->ctx;

    // ending the generation wins: if both are pending there is no point closing the thinking span first
    if (ctx->ask_stop) {
        ctx->ask_stop = false;
        ctx->ask_close_think = false;
        return 2;
    }
    if (ctx->ask_close_think) {
        ctx->ask_close_think = false;
        return 1;
    }
    return 0;
}

common_loop_breaker_state::channel common_loop_breaker_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return common_loop_breaker_state::ANSWER;
    }
    const auto * ctx = (const common_loop_breaker_ctx *) smpl->ctx;
    return ctx->thinking ? common_loop_breaker_state::THINK : common_loop_breaker_state::ANSWER;
}

void common_loop_breaker_set_exempt(struct llama_sampler * smpl, bool exempt) {
    if (!smpl) {
        return;
    }
    auto * ctx = (common_loop_breaker_ctx *) smpl->ctx;
    ctx->exempt = exempt;
}

void common_loop_breaker_get_counters(const struct llama_sampler * smpl,
                                      uint64_t * detections,
                                      uint64_t * nudges,
                                      uint64_t * forced_thinks,
                                      uint64_t * stopped_answers) {
    if (!smpl) {
        return;
    }
    const auto * ctx = (const common_loop_breaker_ctx *) smpl->ctx;

    if (detections)      { *detections      = ctx->st.detections; }
    if (nudges)          { *nudges          = ctx->st.nudges; }
    if (forced_thinks)   { *forced_thinks   = ctx->st.forced_thinks; }
    if (stopped_answers) { *stopped_answers = ctx->st.stopped_answers; }
}
