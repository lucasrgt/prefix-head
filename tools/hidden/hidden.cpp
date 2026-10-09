#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static uint16_t to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    uint32_t h = sign | (exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1) h++;
    return (uint16_t)h;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: hidden <model.gguf> <text> <out-prefix> <ctx> [every] [threads]\n");
        return 1;
    }
    const char * model_path = argv[1];
    int ctx = atoi(argv[4]);
    int every = argc > 5 ? atoi(argv[5]) : 1;
    int threads = argc > 6 ? atoi(argv[6]) : 8;
    std::string out = argv[3];

    std::ifstream f(argv[2]);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int d = llama_model_n_embd(model);
    int nv = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), (int)text.size(), toks.data(), (int)toks.size(), false, false);
    toks.resize(n);
    bool bos = llama_vocab_get_add_bos(vocab);
    int body = bos ? ctx - 1 : ctx;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctx;
    cp.n_batch = ctx;
    cp.n_ubatch = ctx;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * lctx = llama_init_from_model(model, cp);

    FILE * fh = fopen((out + ".h").c_str(), "wb");
    FILE * fy = fopen((out + ".y").c_str(), "wb");
    llama_batch batch = llama_batch_init(ctx, 0, 1);
    std::vector<uint16_t> row(d);
    long rows = 0, chars = 0;
    int windows = (n - 1) / body;
    for (int w = 0; w < windows; w += every) {
        int s = w * body;
        batch.n_tokens = 0;
        auto push = [&](llama_token t) {
            int i = batch.n_tokens++;
            batch.token[i] = t;
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        };
        if (bos) push(llama_vocab_bos(vocab));
        for (int i = 0; i < body; i++) push(toks[s + i]);
        llama_memory_clear(llama_get_memory(lctx), true);
        if (llama_decode(lctx, batch) != 0) {
            fprintf(stderr, "decode failed at window %d\n", w);
            return 1;
        }
        int off = bos ? 1 : 0;
        for (int i = 0; i < body; i++) {
            const float * h = llama_get_embeddings_ith(lctx, off + i);
            for (int k = 0; k < d; k++) row[k] = to_f16(h[k]);
            fwrite(row.data(), 2, d, fh);
            uint32_t y = (uint32_t)toks[s + i + 1];
            fwrite(&y, 4, 1, fy);
            char buf[256];
            int len = llama_token_to_piece(vocab, y, buf, sizeof buf, 0, false);
            for (int b = 0; b < len; b++) chars += ((unsigned char)buf[b] & 0xC0) != 0x80;
            rows++;
        }
        fprintf(stderr, "\rwindow %d/%d", w + 1, windows);
    }
    fclose(fh);
    fclose(fy);

    FILE * fv = fopen((out + ".vocab").c_str(), "wb");
    for (int t = 0; t < nv; t++) {
        char buf[256];
        int len = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
        if (len < 0) len = 0;
        uint8_t l = (uint8_t)len;
        fwrite(&l, 1, 1, fv);
        fwrite(buf, 1, len, fv);
    }
    fclose(fv);
    FILE * fm = fopen((out + ".json").c_str(), "w");
    fprintf(fm, "{\"model\":\"%s\",\"d\":%d,\"vocab\":%d,\"rows\":%ld,\"chars\":%ld,\"ctx\":%d,\"every\":%d,\"tokens\":%d}\n", model_path, d, nv, rows, chars, ctx, every, n);
    fclose(fm);
    fprintf(stderr, "\n%ld rows, %ld chars, d %d, vocab %d\n", rows, chars, d, nv);
    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
