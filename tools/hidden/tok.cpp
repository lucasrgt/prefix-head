#include "llama.h"
#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: tok <model.gguf> < text > ids\n");
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), (int)text.size(), toks.data(), (int)toks.size(), false, false);
    for (int i = 0; i < n; i++) printf("%d\n", toks[i]);
    llama_model_free(model);
    return 0;
}
