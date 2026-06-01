// coconut_qvac — Coconut latent reasoning (continuous thoughts) нативно в llama.cpp/QVAC.
// Между промптом и ответом: K раз подаём last hidden state обратно как ВХОДНОЙ эмбеддинг
// (continuous thought), БЕЗ декодирования токена. Потом генерим ответ. Сравнение K=0 vs K>0.
// Прототип (см. projects/lumi/COCONUT_QVAC.md). ТРЕБУЕТ сборки+проверки.
//   coconut_qvac -m model.gguf -k 2 -n 32 "Вопрос: 2+2=?"
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

static float g_scale = 0.0f;  // >0: L2-норм hidden → ×g_scale перед подачей (--scale)
static int   g_names = 0;     // --names: debug-cb печатает имена тензоров (для поиска target hidden), потом выход
static int   g_lens  = 0;     // --lens: logit-lens превью латентов (проекция hidden→vocab через наш lm_head, топ-k)
static FILE* g_dump  = nullptr;          // --lens-dump FILE: CSV вероятностей цифр 0-9 по шагам (для heatmap-визуализации)
static std::vector<llama_token> g_digit_ids;  // токен-id цифр '0'..'9' (заполняется в main)
static int   g_soft  = 0;     // --soft K: soft-token feedback (Σ p_i·embd top-K вместо сырого hidden); 0=выкл

// logit-lens (tuned-lens-lite): после decode латентного шага логиты УЖЕ посчитаны нашим lm_head (eb.logits=true).
// Печатаем топ-k токенов с softmax-вероятностью = «о чём думает модель в этом латенте». НЕ внешняя модель — наш own head.
// Честно (§5): превью = ПРОЕКЦИЯ латента на токены; латент богаче одного токена (суперпозиция), top-k = «облако смысла».
static void print_lens(struct llama_context * ctx, const struct llama_vocab * vocab, int step, int topk) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; i++) idx[i] = i;
    if (topk > n_vocab) topk = n_vocab;
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
        [&](int a, int b){ return logits[a] > logits[b]; });
    // softmax по топ-k (приблизительно — относительные веса «облака смысла»)
    double mx = logits[idx[0]], sum = 0.0;
    for (int i = 0; i < topk; i++) sum += exp((double)logits[idx[i]] - mx);
    printf("  [латент %d] облако смысла: ", step);
    for (int i = 0; i < topk; i++) {
        char buf[128];
        int n = llama_token_to_piece(vocab, idx[i], buf, sizeof(buf), 0, true);
        std::string tok(buf, n > 0 ? n : 0);
        for (auto & c : tok) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        double p = exp((double)logits[idx[i]] - mx) / (sum > 0 ? sum : 1.0);
        printf("'%s'(%.0f%%) ", tok.c_str(), p * 100.0);
    }
    printf("\n");
    // дамп JSON-строки: top-12 РЕАЛЬНЫХ токенов (слов) + softmax-вероятность = «облако смысла» шага (для виз слов)
    if (g_dump) {
        int kk = 12; if (kk > n_vocab) kk = n_vocab;
        double s2 = 0.0; for (int i = 0; i < kk; i++) s2 += exp((double)logits[idx[i]] - mx);
        fprintf(g_dump, "{\"step\":%d,\"top\":[", step);
        for (int i = 0; i < kk; i++) {
            char buf[128];
            int n = llama_token_to_piece(vocab, idx[i], buf, sizeof(buf), 0, true);
            std::string tok(buf, n > 0 ? n : 0);
            std::string esc;  // JSON-экранирование
            for (char ch : tok) {
                if (ch == '"' || ch == '\\') { esc += '\\'; esc += ch; }
                else if (ch == '\n') esc += "\\n";
                else if (ch == '\r') esc += "\\r";
                else if (ch == '\t') esc += "\\t";
                else if ((unsigned char)ch < 0x20) { }  // пропустить прочие управляющие
                else esc += ch;
            }
            double p = exp((double)logits[idx[i]] - mx) / (s2 > 0 ? s2 : 1.0);
            fprintf(g_dump, "%s[\"%s\",%.4f]", i ? "," : "", esc.c_str(), p);
        }
        fprintf(g_dump, "]}\n");
    }
}
static int   g_nembd = 0;
// debug cb_eval: печатает имена тензоров размера n_embd (кандидаты на hidden последнего слоя)
static bool names_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) return true;
    if (t && t->ne[0] == g_nembd && t->name[0]) printf("TENSOR %s ne=[%lld,%lld]\n", t->name, (long long)t->ne[0], (long long)t->ne[1]);
    return true;
}
// K>0: ловим result_norm (continuous thought, последний hidden перед lm_head) — обновляется на каждый decode.
// + soft-token: при embd_capture ловим входной тензор "embd" (эмбеддинги top-k токенов на throwaway-decode).
struct CocoCB {
    std::vector<float> h; int n_embd; bool got;
    std::vector<float> embd; int embd_k; bool embd_capture; bool embd_got;  // soft-token
};
static bool capture_cb(struct ggml_tensor * t, bool ask, void * ud) {
    if (ask) return true;
    CocoCB * c = (CocoCB*)ud;
    if (!t || !c) return true;
    if (strcmp(t->name, "result_norm") == 0 && t->ne[0] == c->n_embd) {
        // ПОСЛЕДНЯЯ позиция; GPU-тензор → копируем через backend (не t->data напрямую = segfault)
        const size_t off = (size_t)(t->ne[1] - 1) * c->n_embd * sizeof(float);
        ggml_backend_tensor_get(t, c->h.data(), off, (size_t)c->n_embd * sizeof(float));
        c->got = true;
    }
    // soft-token: входной эмбеддинг "embd" (после tok_embd lookup) для всех k токенов throwaway-батча
    if (c->embd_capture && strcmp(t->name, "embd") == 0 && t->ne[0] == c->n_embd && t->ne[1] >= c->embd_k) {
        ggml_backend_tensor_get(t, c->embd.data(), 0, (size_t)c->n_embd * c->embd_k * sizeof(float));
        c->embd_got = true;
    }
    return true;
}

// soft-token feedback: вместо сырого result_norm (OUTPUT-space, OOD) подаём Σ p_i·embd(top-k) — IN-distribution.
// Декодим top-k токены throwaway-батчем (ловим их "embd"), откатываем KV, взвешенно суммируем по softmax-вероятностям.
static bool compute_soft(struct llama_context * ctx, const struct llama_vocab * vocab, CocoCB * c,
                         int n_embd, int n_past, int topk, std::vector<float> & out) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return false;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; i++) idx[i] = i;
    if (topk > n_vocab) topk = n_vocab;
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
        [&](int a, int b){ return logits[a] > logits[b]; });
    double mx = logits[idx[0]], sum = 0.0;
    for (int i = 0; i < topk; i++) sum += exp((double)logits[idx[i]] - mx);
    std::vector<float> probs(topk);
    for (int i = 0; i < topk; i++) probs[i] = (float)(exp((double)logits[idx[i]] - mx) / (sum > 0 ? sum : 1.0));
    // throwaway-decode top-k токенов → callback ловит их "embd"
    c->embd_k = topk; c->embd.assign((size_t)n_embd * topk, 0.0f); c->embd_capture = true; c->embd_got = false;
    llama_batch tb = llama_batch_init(topk, 0, 1);
    for (int i = 0; i < topk; i++) { tb.token[i] = idx[i]; tb.pos[i] = n_past + i; tb.n_seq_id[i] = 1; tb.seq_id[i][0] = 0; tb.logits[i] = false; }
    tb.n_tokens = topk;
    bool ok = (llama_decode(ctx, tb) == 0);
    llama_batch_free(tb);
    c->embd_capture = false;
    llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, n_past + topk);  // откат KV throwaway-позиций
    if (!ok || !c->embd_got) return false;
    out.assign(n_embd, 0.0f);
    for (int i = 0; i < topk; i++)
        for (int j = 0; j < n_embd; j++)
            out[j] += probs[i] * c->embd[(size_t)i * n_embd + j];
    return true;
}

static void usage(const char * a0) {
    printf("\n  %s -m model.gguf [-k K_latent=2] [-n n_predict=32] [-ngl 999] \"prompt\"\n", a0);
}

int main(int argc, char ** argv) {
    std::string model_path, prompt;
    int K = 2, n_predict = 32, ngl = 999;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-k" && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (a == "-n" && i + 1 < argc) n_predict = std::stoi(argv[++i]);
        else if (a == "-ngl" && i + 1 < argc) ngl = std::stoi(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) g_scale = std::stof(argv[++i]);
        else if (a == "--names") g_names = 1;
        else if (a == "--lens") g_lens = 1;
        else if (a == "--lens-dump" && i + 1 < argc) { g_lens = 1; g_dump = fopen(argv[++i], "w"); }
        else if (a == "--soft" && i + 1 < argc) g_soft = std::stoi(argv[++i]);
        else { prompt = a; for (++i; i < argc; i++) { prompt += " "; prompt += argv[i]; } break; }
    }
    if (model_path.empty() || prompt.empty()) { usage(argv[0]); return 1; }
    prompt = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";  // gemma chat-wrap

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load fail\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    // токен-id цифр '0'..'9' для дампа heatmap (берём последний токен токенизации цифры)
    if (g_dump) {
        g_digit_ids.resize(10);
        for (int d = 0; d < 10; d++) {
            std::string ds(1, char('0' + d));
            llama_token tt[4];
            int nt = llama_tokenize(vocab, ds.c_str(), ds.size(), tt, 4, false, false);
            g_digit_ids[d] = nt > 0 ? tt[nt - 1] : 0;
        }
    }

    // tokenize
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true) < 0) {
        fprintf(stderr, "tok fail\n"); return 1;
    }

    // ctx: embeddings=true + pooling NONE → get_embeddings_ith отдаёт per-token hidden (continuous thought)
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_prompt + K + n_predict + 8;
    cp.n_batch = n_prompt + 8;
    cp.embeddings    = false;                    // генерация чистая; hidden берём через callback (не embeddings-режим)
    cp.pooling_type  = LLAMA_POOLING_TYPE_UNSPECIFIED;
    g_nembd = n_embd;
    static CocoCB cbdata; cbdata.h.resize(n_embd); cbdata.n_embd = n_embd; cbdata.got = false;
    cbdata.embd_capture = false; cbdata.embd_got = false; cbdata.embd_k = 0;
    if (g_names)     { cp.cb_eval = names_cb;   cp.cb_eval_user_data = nullptr; }   // debug: печать имён тензоров
    else if (K > 0)  { cp.cb_eval = capture_cb; cp.cb_eval_user_data = &cbdata; }   // K>0: ловить result_norm
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx fail\n"); return 1; }

    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int n_past = 0;
    // 1) decode промпта (явные позиции через batch_init для консистентности с embd-шагами)
    {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            b.token[i] = toks[i]; b.pos[i] = n_past + i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (i == n_prompt - 1);
        }
        b.n_tokens = n_prompt;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode prompt fail\n"); return 1; }
        n_past += n_prompt;
        llama_batch_free(b);
    }
    if (g_names) { llama_free(ctx); llama_model_free(model); return 0; }  // напечатали имена тензоров — выход

    // logit-lens на hidden ПРОМПТА: искренняя «мысль» модели сразу после прочтения вопроса (до латентного дрейфа)
    if (g_lens) { printf("[lens] о чём Lumi думает СРАЗУ после вопроса (искренняя мысль перед латентами):\n"); print_lens(ctx, vocab, 0, 6); }

    // 2) hidden (continuous thought) — callback УЖЕ поймал result_norm последней позиции в cbdata.h
    std::vector<float> h(n_embd);
    if (K > 0) {
        if (!cbdata.got) { fprintf(stderr, "callback не поймал result_norm\n"); return 1; }
        memcpy(h.data(), cbdata.h.data(), n_embd * sizeof(float));
    }

    // 3) K латентных шагов: подаём hidden обратно ВХОДОМ через batch.embd (БЕЗ сэмпла токена)
    for (int k = 0; k < K; k++) {
        // soft-token: вместо сырого result_norm подаём Σ p_i·embd(top-K) — в input-распределении (не OOD)
        if (g_soft > 0) {
            if (!compute_soft(ctx, vocab, &cbdata, n_embd, n_past, g_soft, h)) {
                fprintf(stderr, "compute_soft fail at %d\n", k); return 1;
            }
        }
        // опц. нормализация масштаба hidden под input-эмбеддинг (--scale S: L2-норм → ×S); для soft не нужно
        else if (g_scale > 0.0f) {
            double nrm = 0.0; for (int j = 0; j < n_embd; j++) nrm += (double)h[j]*h[j];
            nrm = nrm > 0 ? 1.0/ (double)sqrt(nrm) : 0.0;
            for (int j = 0; j < n_embd; j++) h[j] = (float)(h[j]*nrm*g_scale);
        }
        llama_batch eb = llama_batch_init(1, n_embd, 1);
        memcpy(eb.embd, h.data(), n_embd * sizeof(float));
        eb.pos[0] = n_past; eb.n_seq_id[0] = 1; eb.seq_id[0][0] = 0; eb.logits[0] = true;
        eb.n_tokens = 1;
        if (llama_decode(ctx, eb)) { fprintf(stderr, "decode latent %d fail\n", k); return 1; }
        n_past++;
        if (g_lens) print_lens(ctx, vocab, k + 1, 6);  // logit-lens: о чём думает модель в этом латенте
        memcpy(h.data(), cbdata.h.data(), n_embd * sizeof(float));  // callback обновил result_norm после decode
        llama_batch_free(eb);
    }

    // 4) генерация ответа токенами от текущего состояния KV
    if (g_lens) printf("[K=%d] (после %d латентных шагов думанья) ", K, K); else printf("[K=%d] ", K);
    for (int t = 0; t < n_predict; t++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) { fwrite(buf, 1, n, stdout); fflush(stdout); }
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = id; b.pos[0] = n_past; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
        b.n_tokens = 1;
        if (llama_decode(ctx, b)) { llama_batch_free(b); break; }
        n_past++;
        llama_batch_free(b);
    }
    printf("\n");

    if (g_dump) fclose(g_dump);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
