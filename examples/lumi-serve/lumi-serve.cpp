// lumi-serve — single resident process: load model ONCE, serve inference AND
// train a memory-LoRA in-process on the same loaded model. No model reloads.
//
// Endpoints (cpp-httplib):
//   GET  /health         -> {"ok":true,"memory":bool}
//   POST /completion     {"prompt","n_predict","temperature"} -> {"content"}
//   POST /train          {"corpus","epochs","rank","lr"} -> trains memory-LoRA,
//                        applies it to the inference context in-process
//   POST /clear_memory   -> drop the applied adapter
//
// Inference ctx: training=false. Training ctx: a second context on the SAME model
// with training=true (model weights shared — NOT reloaded). After training the
// adapter is saved and loaded onto the inference ctx (few MB, no model reload).
//
// Build: target lumi-serve, links llama + common + cpp-httplib (see CMake snippet).
#include "llama.h"
#include "ggml-opt.h"   // ggml_opt_set_loss_scale (GRPO signed-advantage push)
#include "common.h"
#define CPPHTTPLIB_NO_DEFAULT_CONTENT_TYPE
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <mutex>
#include <cstdio>

using json = nlohmann::json;

static llama_model *      g_model   = nullptr;
static llama_context *    g_infer   = nullptr;        // training=false
static const llama_vocab* g_vocab   = nullptr;
static llama_adapter_lora* g_mem    = nullptr;        // applied memory adapter
static std::mutex         g_mutex;                    // inference XOR training
static std::string        g_model_path;
static std::string        g_adapter_path = "/tmp/lumi_memory_inproc.gguf";
static std::string        g_sys  = "";   // weights-only: персона ТОЛЬКО из весов, без промпт-костылей (NO prompt-инжект). Обучалась без system-роли.
static std::string        g_name = "Lumi";
static int                g_ngl  = 999;
static std::vector<std::pair<std::string,std::string>> g_models;  // label -> gguf path (UI switcher)
#include <fstream>
#include <sstream>
#include <map>
// MoE: named online-LoRA experts, applied as a weighted set via llama_set_adapters_lora.
static std::map<std::string, llama_adapter_lora*> g_experts;   // name -> loaded adapter
static std::string g_active_experts = "";                      // for /health display
static std::string expert_path(const std::string & name) { return "/tmp/lumi_mem_" + name + ".gguf"; }

// constant-LR AdamW params for the optimizer
// Built-in single-page UI (chat + consolidate memory). Talks to this same server.
static const char * UI_HTML = R"HTML(<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Lumi edge memory</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#14131a;color:#eee}
h1{font-size:18px;margin:.2em 0}.sub{color:#9b97b0;font-size:13px;margin-bottom:10px}
#chat{border:1px solid #322f40;border-radius:10px;padding:12px;height:50vh;overflow:auto;background:#1c1a24}
.m{margin:8px 0;padding:8px 11px;border-radius:10px;max-width:85%;white-space:pre-wrap;line-height:1.35}
.u{background:#2d3550;margin-left:auto}.l{background:#3a2d44}
.row{display:flex;gap:8px;margin-top:10px}input,button,select{font:inherit;border-radius:8px;border:1px solid #4a4560;background:#221f2c;color:#eee;padding:9px 11px}
#inp{flex:1}button{cursor:pointer}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px;font-size:13px;color:#9b97b0}
.badge{padding:2px 8px;border-radius:20px;font-size:12px}.on{background:#2a5a3a;color:#bfe}.off{background:#3a3340;color:#aaa}
.panel{border:1px solid #322f40;border-radius:10px;padding:8px 11px;margin-top:10px;background:#1a1822}
.panel b{color:#cfc7e0}#mdl{flex:1;min-width:160px}small{color:#6f6b80}</style></head><body>
<h1>Lumi — edge memory</h1><div class="sub">Online‑LoRA: после каждого сообщения модель <b>дообучается в веса</b> в этом процессе (~5с, без перезагрузки). «🔄 новая сессия» чистит контекст, веса остаются — проверь, помнит ли из весов.</div>
<div class="panel"><div class="row"><select id="mdl"></select><button id="load">загрузить модель</button></div>
<small id="curmdl"></small></div>
<div id="chat"></div><div class="row"><input id="inp" placeholder="напиши..." autocomplete="off"><button id="send">→</button></div>
<div class="bar"><label><input type="checkbox" id="auto" checked> online-обучение (учится в веса каждый ход)</label><button id="news">🔄 новая сессия</button><button id="clr">забыть</button>
<span>память: <span id="mem" class="badge off">нет</span></span><span id="st"></span></div>
<div class="panel"><b>модуль памяти (адаптер)</b><div class="row">
<button id="dl">💾 скачать адаптер</button><input type="file" id="upf" accept=".gguf" style="flex:1"><button id="apply">📤 залить + применить</button></div>
<small>скачай обученный адаптер как файл, или залей свой ранее скачанный — он применится в веса поверх модели</small></div>
<script>
let SYS="";
let hist=[];const chat=document.getElementById('chat'),inp=document.getElementById('inp'),st=document.getElementById('st'),mem=document.getElementById('mem');
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
function prompt(u){let p="<start_of_turn>user\n"+(SYS?SYS+"\n\n":"");for(const[r,t]of hist.slice(-6)){p+=r=='u'?t+"<end_of_turn>\n<start_of_turn>model\n":t+"<end_of_turn>\n<start_of_turn>user\n";}return p+u+"<end_of_turn>\n<start_of_turn>model\n";}
async function send(){const t=inp.value.trim();if(!t)return;inp.value='';add(t,'u');const w=add('…','l');
 try{const r=await fetch('/completion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt(t),n_predict:120,temperature:0.4})});
 const j=await r.json();w.textContent=j.content||'(пусто)';hist.push(['u',t]);hist.push(['m',j.content||'']);
 if(document.getElementById('auto').checked)autoTrain();}catch(e){w.textContent='⚠ '+e}}
async function autoTrain(){let c='';for(const[r,t]of hist.slice(-6))c+="<start_of_turn>"+(r=='u'?'user':'model')+"\n"+t+"<end_of_turn>\n";if(!c)return;
 st.textContent='🧠 учусь в веса...';try{const r=await fetch('/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corpus:c,epochs:1,rank:8,lr:0.0004})});
 const j=await r.json();st.textContent=j.ok?'✓ запомнила (в весах)':'⚠ '+(j.error||'fail');poll();}catch(e){st.textContent='⚠ train: '+e}}
document.getElementById('send').onclick=send;inp.addEventListener('keydown',e=>{if(e.key=='Enter')send()});
document.getElementById('news').onclick=()=>{hist=[];chat.innerHTML='';st.textContent='новая сессия (контекст очищен, память в весах осталась)';};
document.getElementById('clr').onclick=async()=>{await fetch('/clear_memory',{method:'POST'});hist=[];chat.innerHTML='';poll();st.textContent='память снята';};
async function loadModels(){try{const j=await(await fetch('/models')).json();const s=document.getElementById('mdl');s.innerHTML='';
 (j.models||[]).forEach(m=>{const o=document.createElement('option');o.value=m.label;o.textContent=m.label+'  ('+m.path.split('/').pop()+')';if(m.path===j.current)o.selected=true;s.appendChild(o);});
 document.getElementById('curmdl').textContent='текущая модель: '+(j.current||'?');}catch(e){}}
document.getElementById('load').onclick=async()=>{const lbl=document.getElementById('mdl').value;st.textContent='⏳ гружу модель '+lbl+'... (модель сменится, память сбросится)';
 try{const r=await fetch('/load_model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:lbl})});const j=await r.json();
 st.textContent=j.ok?('✓ загружена: '+j.model):('⚠ '+(j.error||'fail'));hist=[];chat.innerHTML='';loadModels();poll();}catch(e){st.textContent='⚠ '+e}};
document.getElementById('dl').onclick=()=>{window.location='/adapter/download';};
document.getElementById('apply').onclick=async()=>{const f=document.getElementById('upf').files[0];if(!f){st.textContent='выбери .gguf файл';return;}
 st.textContent='📤 заливаю адаптер '+f.name+'...';try{const buf=await f.arrayBuffer();const r=await fetch('/adapter/upload',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:buf});const j=await r.json();
 st.textContent=j.ok?('✓ адаптер применён ('+j.bytes+' байт)'):('⚠ '+(j.error||'fail'));poll();}catch(e){st.textContent='⚠ '+e}};
async function poll(){try{const j=await(await fetch('/health')).json();if(j.sys!==undefined)SYS=j.sys;if(j.name){document.querySelector('h1').textContent=j.name+' — edge memory';document.title=j.name;}mem.textContent=j.memory?'в весах ✓':'нет';mem.className='badge '+(j.memory?'on':'off');}catch(e){}}
loadModels();setInterval(poll,4000);poll();
</script></body></html>)HTML";

static float g_lr = 2e-4f;
static ggml_opt_optimizer_params opt_pars_cb(void * ud) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.adamw.alpha = *(float*)ud;
    p.adamw.wd    = 0.0f;
    return p;
}

static std::string generate(const std::string & prompt, int n_predict, float temp, int n_latent=0) {
    // stateless per request: clear KV, decode prompt, sample n_predict tokens.
    // n_latent>0: Coconut-style латентная мультипроходность — после промпта переподаём last hidden
    // state как input-embedding K раз («думает внутри», без эмита токенов), потом декодим ответ.
    llama_memory_clear(llama_get_memory(g_infer), true);
    llama_set_embeddings(g_infer, n_latent > 0);
    const int n_prompt = -llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> toks(n_prompt);
    llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // repetition penalty FIRST — without it the small model loops on the persona spiel
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp <= 0 ? 0.01f : temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string out;
    const int n_embd = llama_model_n_embd(g_model);
    int npos = (int) toks.size();
    // decode prompt
    llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
    if (llama_decode(g_infer, batch) != 0) { llama_sampler_free(smpl); return out; }

    // === Coconut латентная мультипроходность: переподача last hidden как input-embd K раз ===
    if (n_latent > 0) {
        llama_batch lb = llama_batch_init(1, n_embd, 1);
        std::vector<float> hbuf(n_embd);
        for (int k = 0; k < n_latent; ++k) {
            const float * h = llama_get_embeddings_ith(g_infer, -1);
            if (!h) break;
            memcpy(hbuf.data(), h, n_embd * sizeof(float));
            lb.n_tokens = 1;
            memcpy(lb.embd, hbuf.data(), n_embd * sizeof(float));
            lb.pos[0] = npos++; lb.n_seq_id[0] = 1; lb.seq_id[0][0] = 0; lb.logits[0] = 1;
            if (llama_decode(g_infer, lb) != 0) break;
        }
        llama_batch_free(lb);
    }

    // === обычная генерация токенов (KV уже содержит промпт + латентные шаги) ===
    for (int i = 0; i < n_predict; ++i) {
        llama_token id = llama_sampler_sample(smpl, g_infer, -1);
        if (llama_vocab_is_eog(g_vocab, id)) break;
        char buf[256];
        int n = llama_token_to_piece(g_vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);
        static llama_token one; one = id;
        batch = llama_batch_get_one(&one, 1);
        if ((int) out.size() > 4000) break;
        if (llama_decode(g_infer, batch) != 0) break;
    }
    llama_sampler_free(smpl);
    // trim gemma turn markers
    for (const char * s : {"<end_of_turn>", "<start_of_turn>"}) {
        auto p = out.find(s); if (p != std::string::npos) out.resize(p);
    }
    return out;
}

// load a LoRA adapter file and apply it to the inference ctx (no model reload).
static bool apply_adapter_file(const std::string & path, std::string & err) {
    llama_adapter_lora * a = llama_adapter_lora_init(g_model, path.c_str());
    if (!a) { err = "adapter init failed (wrong file / arch mismatch?)"; return false; }
    g_mem = a;
    llama_adapter_lora * arr[1] = { g_mem }; float sc[1] = { 1.0f };
    llama_set_adapters_lora(g_infer, arr, 1, sc);
    return true;
}

// (re)load a model from disk, freeing the previous one first (VRAM-safe: never two
// models resident at once). Recreates the inference ctx. Drops any applied adapter.
static bool load_model_into(const std::string & path, std::string & err) {
    if (g_infer) { llama_free(g_infer); g_infer = nullptr; }
    g_mem = nullptr;  // adapter belonged to the old model; freed with it
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_ngl;
    g_model = llama_model_load_from_file(path.c_str(), mp);
    if (!g_model) { err = "model load failed: " + path; return false; }
    g_vocab = llama_model_get_vocab(g_model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    g_infer = llama_init_from_model(g_model, cp);
    if (!g_infer) { err = "ctx init failed"; return false; }
    g_model_path = path;
    return true;
}

static bool train_memory(const std::string & corpus, int epochs, int rank, float lr, std::string & err,
                         const std::string & save_path, int targets, float reward, bool assistant_only=false) {
    // assistant_only: corpus = JSONL conversations ({"messages":[...]}); loss masked to ASSISTANT
    // tokens only (prompt = no loss). Без этого CE по всему корпусу тонет в шуме промпта → не учит.
    // Count corpus tokens BEFORE creating the ctx, to size n_ctx adaptively. The old fixed
    // n_ctx=512 chopped multi-example RL corpora into 512-tok windows ACROSS example
    // boundaries → trained on incoherent spans → collapse (STaR). Fit the corpus in one
    // window (cap 2048) so examples stay coherent → stable multi-example self-training.
    int n_corpus = -llama_tokenize(g_vocab, corpus.c_str(), corpus.size(), nullptr, 0, true, true);
    if (n_corpus < 1) { err = "empty corpus"; return false; }
    // PROVEN-SAFE fixed 512 (n_ctx=n_batch=n_ubatch). The earlier "adaptive n_ctx→2048" blew
    // VRAM 6→24GB+shared (training activations scale with ubatch); n_ubatch!=n_ctx crashes the
    // opt graph. So: keep all three = 512 (this config trained fine and gave +7pp, ~6-11GB peak).
    const int nctx = 512;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nctx; cp.n_batch = nctx; cp.n_ubatch = nctx;
    cp.training = true;            // <-- enables LoRA gradient flow
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * tctx = llama_init_from_model(g_model, cp);
    if (!tctx) { err = "failed to create training context"; return false; }

    bool ok = false;
    llama_lora_training_params lp{};
    lp.target_modules = targets ? targets : (LLAMA_LORA_TARGET_ATTN_Q | LLAMA_LORA_TARGET_ATTN_V);
    lp.rank = rank; lp.alpha = rank * 2.0f; lp.dropout = 0.0f; lp.init_std = 0.02f; lp.seed = 0;
    llama_adapter_lora * adapter = llama_lora_training_init(tctx, g_model, &lp);
    if (!adapter) { err = "lora_training_init failed"; llama_free(tctx); return false; }

    // dataset prep underflows if corpus < n_ctx (ndata = n_tokens - n_ctx wraps to ~2^64).
    // Repeat ONLY short corpora to exceed n_ctx; long (multi-example) corpora are used as-is.
    ggml_opt_dataset_t ds = nullptr;
    if (assistant_only) {
        // masked SFT: лосс только на assistant-токены (chat-template модели). corpus = JSONL convs.
        ds = common_opt_sft_dataset_init(tctx, corpus, llama_n_ctx(tctx)/2, "");
    } else {
        std::string corpus_rep = corpus;
        if ((size_t) n_corpus < (size_t) nctx * 2) {
            size_t reps = (((size_t) nctx * 2) / (size_t) n_corpus) + 2;
            corpus_rep.clear();
            for (size_t i = 0; i < reps; ++i) corpus_rep += corpus;
        }
        std::vector<llama_token> tokens = common_tokenize(tctx, corpus_rep, true);
        ds = common_opt_dataset_init(tctx, tokens, llama_n_ctx(tctx)/2);
    }
    if (!ds) { err = "dataset init failed"; llama_free(tctx); return false; }

    // reward-weighted step (REINFORCE/GRPO-lite): scale lr by reward (signed advantage).
    // reward>0 → descend CE on the corpus (reinforce). reward<0 → ASCEND CE (unlearn wrong
    // samples = GRPO negative push). Clamp |lr| to SAFE range (|lr|>3e-4 → collapse, проверено).
    // The controller passes correct samples with reward>0 and wrong samples with reward<0.
    // alpha (lr) MUST stay > 0 (AdamW asserts alpha>0). Put the SIGNED advantage into the loss
    // scale (GRPO negative push): reward>0 → descend (reinforce), reward<0 → ascend (unlearn wrong).
    g_lr = lr; if (g_lr > 3e-4f) g_lr = 3e-4f; if (g_lr < 1e-6f) g_lr = 1e-6f;
    float adv = reward; if (adv > 2.0f) adv = 2.0f; if (adv < -1.0f) adv = -1.0f;  // bound
    ggml_opt_set_loss_scale(adv);
    llama_opt_params op = llama_opt_default_params();
    op.param_filter = llama_opt_param_filter_lora;
    op.get_opt_pars = opt_pars_cb; op.get_opt_pars_ud = &g_lr;
    op.assistant_loss_only = assistant_only;   // masked CE на assistant-токены
    llama_opt_init(tctx, g_model, op);

    int64_t ndata = ggml_opt_dataset_ndata(ds);
    int64_t split = (int64_t)(ndata * 0.95);
    if (split < 1) split = ndata;
    ggml_opt_result_t rt = ggml_opt_result_init(), re = ggml_opt_result_init();
    for (int e = 0; e < epochs; ++e) {
        llama_opt_epoch(tctx, ds, rt, re, split, nullptr, nullptr);
        ggml_opt_result_reset(rt); ggml_opt_result_reset(re);
    }
    ggml_opt_result_free(rt); ggml_opt_result_free(re);
    ggml_opt_set_loss_scale(1.0f);   // reset (default identity) for next call

    if (llama_lora_save_adapter(adapter, save_path.c_str(), g_model)) ok = true;
    else err = "save_adapter failed";
    llama_free(tctx);   // free training buffers; model stays resident

    if (ok) {
        // apply to the inference context (load the few-MB adapter — no model reload)
        if (!apply_adapter_file(save_path, err)) ok = false;
    }
    return ok;
}

int main(int argc, char ** argv) {
    std::string host = "0.0.0.0"; int port = 8770; int ngl = 999;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i+1 < argc) g_model_path = argv[++i];
        else if (a == "--port" && i+1 < argc) port = atoi(argv[++i]);
        else if (a == "--host" && i+1 < argc) host = argv[++i];
        else if (a == "-ngl" && i+1 < argc) ngl = atoi(argv[++i]);
        else if (a == "--sys" && i+1 < argc) g_sys = argv[++i];
        else if (a == "--name" && i+1 < argc) g_name = argv[++i];
        // --models "Lumi=/path/lumi.gguf,Base=/path/base.gguf" — UI model switcher
        else if (a == "--models" && i+1 < argc) {
            std::string spec = argv[++i]; std::stringstream ss(spec); std::string item;
            while (std::getline(ss, item, ',')) {
                auto eq = item.find('=');
                if (eq != std::string::npos) g_models.push_back({item.substr(0,eq), item.substr(eq+1)});
            }
        }
    }
    g_ngl = ngl;
    if (g_model_path.empty() && !g_models.empty()) g_model_path = g_models.front().second;
    // ensure the launch model is in the switcher list
    if (!g_model_path.empty()) {
        bool found = false; for (auto & m : g_models) if (m.second == g_model_path) found = true;
        if (!found) g_models.insert(g_models.begin(), {g_name, g_model_path});
    }
    llama_backend_init();
    ggml_backend_load_all();
    std::string lerr;
    if (!load_model_into(g_model_path, lerr)) { fprintf(stderr, "%s\n", lerr.c_str()); return 1; }
    fprintf(stderr, "lumi-serve: model resident, listening on %s:%d\n", host.c_str(), port);

    httplib::Server srv;
    srv.set_payload_max_length(256ull * 1024 * 1024);  // adapter uploads are a few MB
    srv.Post("/completion", [](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { res.status = 400; return; }
        std::lock_guard<std::mutex> lk(g_mutex);
        std::string out = generate(j.value("prompt", ""), j.value("n_predict", 120), j.value("temperature", 0.4f), j.value("n_latent", 0));
        res.set_content(json{{"content", out}}.dump(), "application/json");
    });
    srv.Post("/train", [](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { res.status = 400; return; }
        std::lock_guard<std::mutex> lk(g_mutex);
        std::string err;
        // optional "name" → save as a named MoE expert (/tmp/lumi_mem_<name>.gguf)
        std::string name = j.value("name", "");
        std::string save = name.empty() ? g_adapter_path : expert_path(name);
        bool ok = train_memory(j.value("corpus", ""), j.value("epochs", 2),
                               j.value("rank", 8), j.value("lr", 2e-4f), err, save,
                               j.value("targets", 0),       // 0=default(Q|V); bitmask per llama.h
                               j.value("reward", 1.0f),     // reward-weighted step (GRPO/REINFORCE)
                               j.value("assistant_only", false)); // masked CE на assistant-токены (corpus=JSONL convs)
        res.set_content(json{{"ok", ok}, {"error", err}, {"name", name}, {"path", save}}.dump(), "application/json");
    });
    // MoE: apply a weighted set of named online-LoRA experts simultaneously.
    // body: {"experts":[{"name":"cats","scale":1.0},{"name":"coffee","scale":0.5}]}
    srv.Post("/moe", [](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded() || !j.contains("experts")) { res.status = 400; return; }
        std::lock_guard<std::mutex> lk(g_mutex);
        std::vector<llama_adapter_lora*> arr; std::vector<float> sc; std::string active, err;
        for (auto & e : j["experts"]) {
            std::string name = e.value("name", ""); float scale = e.value("scale", 1.0f);
            if (name.empty()) continue;
            auto it = g_experts.find(name);
            llama_adapter_lora * a = (it != g_experts.end()) ? it->second : nullptr;
            if (!a) {  // lazy-load from disk
                a = llama_adapter_lora_init(g_model, expert_path(name).c_str());
                if (!a) { err += "load fail: " + name + "; "; continue; }
                g_experts[name] = a;
            }
            arr.push_back(a); sc.push_back(scale);
            active += name + "(" + std::to_string(scale).substr(0,4) + ") ";
        }
        if (arr.empty()) { res.set_content(json{{"ok",false},{"error","no experts loaded; "+err}}.dump(), "application/json"); return; }
        llama_set_adapters_lora(g_infer, arr.data(), arr.size(), sc.data());
        g_mem = arr[0]; g_active_experts = active;            // mark memory active
        res.set_content(json{{"ok",true},{"active",active},{"error",err},{"count",(int)arr.size()}}.dump(), "application/json");
    });
    srv.Post("/clear_memory", [](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (g_mem) { llama_set_adapters_lora(g_infer, nullptr, 0, nullptr); g_mem = nullptr; }
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"ok", true}, {"memory", g_mem != nullptr}, {"name", g_name},
                             {"sys", g_sys}, {"model", g_model_path}, {"experts", g_active_experts}}.dump(), "application/json");
    });
    // list of switchable models (label/path) + currently loaded one
    srv.Get("/models", [](const httplib::Request &, httplib::Response & res) {
        json arr = json::array();
        for (auto & m : g_models) arr.push_back({{"label", m.first}, {"path", m.second}});
        res.set_content(json{{"models", arr}, {"current", g_model_path}}.dump(), "application/json");
    });
    // switch the resident model (by label or explicit path). VRAM-safe: frees old first.
    srv.Post("/load_model", [](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { res.status = 400; return; }
        std::string path = j.value("path", "");
        std::string label = j.value("label", "");
        if (path.empty() && !label.empty())
            for (auto & m : g_models) if (m.first == label) path = m.second;
        if (path.empty()) { res.set_content(json{{"ok",false},{"error","no path/label"}}.dump(), "application/json"); return; }
        std::lock_guard<std::mutex> lk(g_mutex);
        std::string err; bool ok = load_model_into(path, err);
        res.set_content(json{{"ok", ok}, {"error", err}, {"model", g_model_path}}.dump(), "application/json");
    });
    // download the current memory-adapter file (the "memory module")
    srv.Get("/adapter/download", [](const httplib::Request &, httplib::Response & res) {
        std::ifstream f(g_adapter_path, std::ios::binary);
        if (!f) { res.status = 404; res.set_content("no adapter yet", "text/plain"); return; }
        std::stringstream ss; ss << f.rdbuf();
        res.set_header("Content-Disposition", "attachment; filename=\"lumi_memory.gguf\"");
        res.set_content(ss.str(), "application/octet-stream");
    });
    // upload a memory-adapter file (raw gguf bytes in body) and apply it in-process
    srv.Post("/adapter/upload", [](const httplib::Request & req, httplib::Response & res) {
        if (req.body.size() < 16) { res.set_content(json{{"ok",false},{"error","empty body"}}.dump(), "application/json"); return; }
        std::lock_guard<std::mutex> lk(g_mutex);
        { std::ofstream f(g_adapter_path, std::ios::binary); f.write(req.body.data(), req.body.size()); }
        std::string err; bool ok = apply_adapter_file(g_adapter_path, err);
        res.set_content(json{{"ok", ok}, {"error", err}, {"bytes", (int)req.body.size()}}.dump(), "application/json");
    });
    srv.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(UI_HTML, "text/html; charset=utf-8");
    });
    srv.listen(host, port);
    return 0;
}
