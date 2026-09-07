#include "generation.h"

#include "api_error.h"

#include "iian/log.h"
#include "iian/tokenizer.h"

#include <stdexcept>

namespace iian::server {

GenerationRun::GenerationRun(ServerContext & ctx, GenerationRequest req) : ctx_(ctx), req_(std::move(req)) {
    const size_t total = req_.total_choices();
    choices_.resize(total);
    results_.resize(total);
    for (size_t pi = 0, k = 0; pi < req_.prompts.size(); pi++) {
        for (int i = 0; i < req_.n; i++, k++) {
            results_[k].index = req_.kind == GenerationRequest::Kind::CHAT ? i : (int) k;   // completions: index over prompts*n
            results_[k].prompt_index = pi;
            results_[k].n_prompt = (uint32_t) req_.prompts[pi].size();
        }
    }
    ctx_.counters.in_flight++;
}

GenerationRun::~GenerationRun() {
    abort_all("connection closed");
    record();
    ctx_.active.remove(req_.id);
    ctx_.counters.in_flight--;
}

void GenerationRun::submit() {
    Engine & engine = ctx_.require_engine();
    for (size_t k = 0; k < choices_.size(); k++) {
        SamplingParams p = req_.params;
        if (p.seed && req_.n > 1) p.seed = *p.seed + (uint64_t) (k % (size_t) req_.n);
        RequestOptions o;
        o.external_id = req_.id;
        o.priority = req_.priority;
        o.cache_salt = req_.cache_salt;
        try {
            choices_[k].handle = engine.submit(req_.prompts[results_[k].prompt_index], std::move(p), std::move(o));
            choices_[k].submitted = true;
            ctx_.active.add(req_.id, choices_[k].handle.id);
        } catch (const std::invalid_argument & e) {
            abort_all("sibling choice rejected");
            throw ApiError::bad_request(e.what());
        } catch (const std::exception & e) {
            abort_all("sibling choice rejected");
            throw ApiError::internal(std::string("engine rejected the request: ") + e.what());
        }
    }
    LOG_DBG("http", "%s: submitted %zu choice(s), %u prompt tokens", req_.id.c_str(), choices_.size(), results_.empty() ? 0u : results_[0].n_prompt);
}

void GenerationRun::abort_all(const char * why) {
    Engine * engine = ctx_.engine.load();
    bool any = false;
    for (auto & c : choices_) {
        if (c.submitted && !c.finished) {
            if (engine) engine->abort(c.handle.id);
            c.finished = true;
            any = true;
        }
    }
    if (any) {
        aborted_ = true;
        for (auto & r : results_) if (r.finish == FinishReason::NONE) r.finish = FinishReason::ABORT;
        LOG_INF("http", "%s: aborted (%s)", req_.id.c_str(), why);
    }
}

bool GenerationRun::all_finished() const {
    for (const auto & c : choices_) if (!c.finished) return false;
    return true;
}

void GenerationRun::apply(size_t ci, const OutputChunk & c) {
    ChoiceResult & r = results_[ci];
    r.text += c.text;
    r.tokens.insert(r.tokens.end(), c.tokens.begin(), c.tokens.end());
    r.logprobs.insert(r.logprobs.end(), c.logprobs.begin(), c.logprobs.end());
    if (c.n_prompt_tokens) r.n_prompt = c.n_prompt_tokens;
    r.n_output = c.n_output_tokens ? c.n_output_tokens : (uint32_t) r.tokens.size();
    if (c.finished) {
        r.finish = c.finish_reason;
        r.stop_reason = c.stop_reason;
        r.error = c.error;
        r.n_cached = c.n_cached_tokens;
        choices_[ci].finished = true;
    }
}

std::vector<GenerationRun::Event> GenerationRun::pump(std::chrono::milliseconds wait) {
    std::vector<Event> ev;
    // drain whatever is ready
    for (size_t k = 0; k < choices_.size(); k++) {
        if (choices_[k].finished || !choices_[k].submitted) continue;
        OutputChunk c;
        while (!choices_[k].finished && choices_[k].handle.out->try_pop(c)) { apply(k, c); ev.push_back({k, std::move(c)}); c = OutputChunk{}; }
    }
    if (!ev.empty()) return ev;
    // nothing ready: block on the first unfinished queue
    for (size_t k = 0; k < choices_.size(); k++) {
        if (choices_[k].finished || !choices_[k].submitted) continue;
        OutputChunk c;
        if (choices_[k].handle.out->pop(c, wait)) { apply(k, c); ev.push_back({k, std::move(c)}); }
        break;
    }
    return ev;
}

bool GenerationRun::collect(const std::function<bool()> & client_gone) {
    while (!all_finished()) {
        pump(std::chrono::milliseconds(100));
        if (all_finished()) break;
        if (ctx_.stopping.load()) { abort_all("server shutting down"); return false; }
        if (client_gone && client_gone()) { abort_all("client disconnected"); return false; }
    }
    return !aborted_;
}

uint64_t GenerationRun::prompt_tokens() const {
    uint64_t n = 0;
    std::vector<bool> seen(req_.prompts.size(), false);
    for (const auto & r : results_) if (!seen[r.prompt_index]) { seen[r.prompt_index] = true; n += r.n_prompt; }
    return n;
}
uint64_t GenerationRun::completion_tokens() const { uint64_t n = 0; for (const auto & r : results_) n += r.n_output; return n; }
uint64_t GenerationRun::cached_tokens() const {
    uint64_t n = 0;
    std::vector<bool> seen(req_.prompts.size(), false);
    for (const auto & r : results_) if (!seen[r.prompt_index]) { seen[r.prompt_index] = true; n += r.n_cached; }
    return n;
}

void GenerationRun::record() {
    if (recorded_) return;
    recorded_ = true;
    for (const auto & r : results_) {
        switch (r.finish) {
            case FinishReason::STOP: ctx_.counters.finished_stop++; break;
            case FinishReason::LENGTH: ctx_.counters.finished_length++; break;
            case FinishReason::ERROR: ctx_.counters.finished_error++; break;
            default: ctx_.counters.finished_abort++; break;
        }
    }
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start_).count();
    ctx_.counters.e2e_latency_us_sum += (uint64_t) us;
    ctx_.counters.e2e_latency_count++;
}

json GenerationRun::response_json() {
    for (const auto & r : results_) {
        if (r.finish == FinishReason::ERROR) throw ApiError::internal("generation failed: " + (r.error.empty() ? std::string("unknown engine error") : r.error));
        if (r.finish == FinishReason::ABORT) throw ApiError::unavailable("request was aborted" + (r.error.empty() ? std::string("") : ": " + r.error));
    }
    return req_.kind == GenerationRequest::Kind::CHAT ? chat_response_json(ctx_, req_, results_) : completion_response_json(ctx_, req_, results_);
}

// ---- streaming ---------------------------------------------------------------------------------

std::string GenerationRun::sse_for(size_t ci, const OutputChunk & c) {
    const Tokenizer & tok = ctx_.require_engine().model().tokenizer();
    Choice & ch = choices_[ci];
    ChoiceResult & r = results_[ci];
    std::string out;
    if (!c.error.empty()) {
        ApiError err(500, "generation failed: " + c.error, "internal_error");
        out += sse_frame(err.to_json());
        return out;
    }
    const bool is_chat = req_.kind == GenerationRequest::Kind::CHAT;
    json lp = nullptr;
    if (req_.want_logprobs && !c.logprobs.empty()) {
        if (is_chat) lp = chat_logprobs_json(tok, c.logprobs, req_.top_logprobs);
        else {
            size_t base = req_.echo ? req_.prompt_texts[r.prompt_index].size() : 0;
            lp = completion_logprobs_json(tok, c.logprobs, req_.top_logprobs, base, ch.lp_offset);
        }
    }
    if (is_chat) {
        if (!ch.role_sent) {
            ch.role_sent = true;
            out += sse_frame(chat_chunk_json(req_, r.index, json{{"role", "assistant"}, {"content", ""}}, nullptr, FinishReason::NONE, ""));
        }
        if (!req_.parse_tool_calls) {
            if (!c.text.empty() || !lp.is_null())
                out += sse_frame(chat_chunk_json(req_, r.index, json{{"content", c.text}}, lp, FinishReason::NONE, ""));
            if (c.finished)
                out += sse_frame(chat_chunk_json(req_, r.index, json::object(), nullptr, c.finish_reason, c.stop_reason));
        } else {
            // tools: stream content only up to where a tool call might start; parse the rest on completion
            // (r.text already contains c.text: apply() ran before sse_for())
            if (!c.finished) {
                size_t safe = streamable_prefix(r.text, req_.tool_format);
                if (safe > ch.content_sent) {
                    out += sse_frame(chat_chunk_json(req_, r.index, json{{"content", r.text.substr(ch.content_sent, safe - ch.content_sent)}}, lp, FinishReason::NONE, ""));
                    ch.content_sent = safe;
                } else if (!lp.is_null()) {
                    out += sse_frame(chat_chunk_json(req_, r.index, json::object(), lp, FinishReason::NONE, ""));
                }
            } else {
                ParsedAssistant pa = parse_tool_calls(r.text, req_.tool_format);
                FinishReason fin = c.finish_reason;
                if (pa.tool_calls.empty()) {
                    if (r.text.size() > ch.content_sent)
                        out += sse_frame(chat_chunk_json(req_, r.index, json{{"content", r.text.substr(ch.content_sent)}}, lp, FinishReason::NONE, ""));
                } else {
                    // content that precedes the first call and was not streamed yet
                    std::string sent = r.text.substr(0, ch.content_sent);
                    if (pa.content.size() > sent.size() && pa.content.compare(0, sent.size(), sent) == 0)
                        out += sse_frame(chat_chunk_json(req_, r.index, json{{"content", pa.content.substr(sent.size())}}, nullptr, FinishReason::NONE, ""));
                    out += sse_frame(chat_chunk_json(req_, r.index, json{{"tool_calls", tool_calls_delta_json(pa.tool_calls)}}, lp, FinishReason::NONE, ""));
                    if (fin == FinishReason::STOP) fin = FinishReason::TOOL_CALLS;
                }
                ch.content_sent = r.text.size();
                out += sse_frame(chat_chunk_json(req_, r.index, json::object(), nullptr, fin, c.stop_reason));
            }
        }
    } else {
        std::string text = c.text;
        if (req_.echo && !ch.echo_sent) { ch.echo_sent = true; text = req_.prompt_texts[r.prompt_index] + text; }
        if (!text.empty() || !lp.is_null() || c.finished)
            out += sse_frame(completion_chunk_json(req_, r.index, text, lp, c.finish_reason, c.stop_reason));
    }
    return out;
}

bool GenerationRun::next_sse(std::string & out, std::chrono::milliseconds wait) {
    if (done_sent_) return false;
    if (!all_finished()) {
        for (auto & e : pump(wait)) out += sse_for(e.choice, e.chunk);
    }
    if (all_finished()) {
        if (aborted_) {
            // choices we aborted ourselves never produced a finished chunk: close them out
            for (size_t k = 0; k < choices_.size(); k++) {
                if (req_.kind == GenerationRequest::Kind::CHAT) {
                    if (results_[k].finish == FinishReason::ABORT && choices_[k].role_sent)
                        out += sse_frame(chat_chunk_json(req_, results_[k].index, json::object(), nullptr, FinishReason::ABORT, ""));
                }
            }
        }
        if (req_.include_usage) out += sse_frame(usage_chunk_json(req_, prompt_tokens(), completion_tokens(), cached_tokens()));
        out += "data: [DONE]\n\n";
        done_sent_ = true;
        return false;
    }
    return true;
}

} // namespace iian::server
